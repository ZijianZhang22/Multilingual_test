import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def accuracy(logits, labels):
    return (logits.argmax(dim=-1) == labels).float().mean().item()


def irm_penalty(logits, labels):
    scale = torch.tensor(1.0, device=logits.device, requires_grad=True)
    loss = F.cross_entropy(logits * scale, labels)
    grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
    return grad.pow(2)


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, strength):
        ctx.strength = strength
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.strength * grad_output, None


class ProjectionTaskModel(nn.Module):
    def __init__(self, input_dim, proj_dim, n_classes):
        super().__init__()
        self.proj = nn.Linear(input_dim, proj_dim, bias=False)
        self.task_head = nn.Linear(proj_dim, n_classes)

    def encode(self, x):
        z = self.proj(x)
        return F.layer_norm(z, (z.shape[-1],))

    def forward(self, x):
        z = self.encode(x)
        return z, self.task_head(z)


class DANNModel(ProjectionTaskModel):
    def __init__(self, input_dim, proj_dim, n_classes, n_domains):
        super().__init__(input_dim, proj_dim, n_classes)
        self.lang_head = nn.Linear(proj_dim, n_domains)

    def forward_with_domain(self, x, grl_strength):
        z = self.encode(x)
        task_logits = self.task_head(z)
        rev = GradReverse.apply(z, grl_strength)
        lang_logits = self.lang_head(rev)
        return z, task_logits, lang_logits


def make_split_indices(splits, split_name):
    return torch.tensor(
        [i for i, s in enumerate(splits) if s == split_name],
        dtype=torch.long,
    )


def train_linear_classifier(x, y, n_classes, epochs, lr, weight_decay, seed):
    set_seed(seed)
    head = nn.Linear(x.shape[1], n_classes).to(x.device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(head(x.detach()), y)
        loss.backward()
        opt.step()
    return head


def fit_projection_method(
    method,
    x,
    y,
    env_y,
    train_idx,
    *,
    proj_dim,
    epochs,
    lr,
    weight_decay,
    irm_lambda,
    vrex_lambda,
    dann_lambda,
    seed,
):
    n_classes = int(y.max().item()) + 1
    train_envs = sorted(set(env_y[train_idx].tolist()))

    if method == "dann":
        env_to_local = {e: i for i, e in enumerate(train_envs)}
        local_env_y = torch.tensor(
            [env_to_local.get(int(e), -1) for e in env_y.tolist()],
            device=x.device,
            dtype=torch.long,
        )
        model = DANNModel(
            x.shape[1], proj_dim, n_classes, len(train_envs)
        ).to(x.device)
    else:
        model = ProjectionTaskModel(
            x.shape[1], proj_dim, n_classes
        ).to(x.device)

    set_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for _ in range(epochs):
        model.train()
        opt.zero_grad(set_to_none=True)

        if method == "dann":
            _, task_logits, lang_logits = model.forward_with_domain(
                x[train_idx], dann_lambda
            )
            loss = (
                F.cross_entropy(task_logits, y[train_idx])
                + F.cross_entropy(lang_logits, local_env_y[train_idx])
            )
        else:
            env_losses = []
            env_penalties = []
            for env_id in train_envs:
                idx = train_idx[env_y[train_idx] == env_id]
                _, logits = model(x[idx])
                env_losses.append(F.cross_entropy(logits, y[idx]))
                if method == "irm":
                    env_penalties.append(irm_penalty(logits, y[idx]))

            losses = torch.stack(env_losses)
            erm = losses.mean()
            if method == "erm":
                loss = erm
            elif method == "irm":
                loss = erm + irm_lambda * torch.stack(env_penalties).mean()
            elif method == "vrex":
                loss = erm + vrex_lambda * losses.var(unbiased=False)
            else:
                raise ValueError(method)

        loss.backward()
        opt.step()

    return model


def fit_inlp(
    x,
    env_y,
    train_idx,
    *,
    iters,
    classifier_epochs,
    classifier_lr,
    weight_decay,
    seed,
):
    """Iteratively erase linearly decodable language directions.

    The representation dimensionality is kept unchanged. Each round trains a
    linear language classifier and projects the representation onto the
    classifier row-space nullspace.
    """
    z = x.clone()
    removed = []
    n_domains = len(set(env_y[train_idx].tolist()))

    for step in range(iters):
        head = train_linear_classifier(
            z[train_idx],
            env_y[train_idx],
            n_domains,
            classifier_epochs,
            classifier_lr,
            weight_decay,
            seed + step,
        )
        with torch.no_grad():
            w = head.weight.detach()
            # Softmax weights have an irrelevant shared offset. Centering
            # isolates between-language discriminative directions.
            w = w - w.mean(dim=0, keepdim=True)
            # Orthonormal basis for the row-space of W.
            _, s, vh = torch.linalg.svd(w, full_matrices=False)
            rank = int((s > 1e-6).sum().item())
            if rank == 0:
                break
            q = vh[:rank].T
            # q is expressed in the original feature coordinates because z
            # retains the original dimensionality after each projection.
            z = z - (z @ q) @ q.T
            removed.append(q.cpu())

    if removed:
        q_total = torch.cat(removed, dim=1).to(x.device)
        q_total, _ = torch.linalg.qr(q_total, mode="reduced")
        z = x - (x @ q_total) @ q_total.T
    else:
        q_total = torch.empty(x.shape[1], 0, device=x.device)

    return z, q_total


def evaluate_task_by_language(task_logits, labels, language_ids, indices, id_to_lang):
    rows = []
    losses = []
    accs = []
    for lang_id, lang in id_to_lang.items():
        idx = indices[language_ids[indices] == lang_id]
        if len(idx) == 0:
            continue
        loss = F.cross_entropy(task_logits[idx], labels[idx]).item()
        acc = accuracy(task_logits[idx], labels[idx])
        rows.append((lang, loss, acc, len(idx)))
        losses.append(loss)
        accs.append(acc)
    return rows, float(np.mean(losses)), float(np.var(losses)), float(np.mean(accs))


def fresh_language_probe_accuracy(
    z,
    language_ids,
    train_idx,
    test_idx,
    n_langs,
    *,
    epochs,
    lr,
    weight_decay,
    seed,
):
    head = train_linear_classifier(
        z[train_idx],
        language_ids[train_idx],
        n_langs,
        epochs,
        lr,
        weight_decay,
        seed,
    )
    with torch.no_grad():
        return accuracy(head(z[test_idx]), language_ids[test_idx])


def fit_task_head_on_representation(
    z,
    labels,
    train_idx,
    *,
    epochs,
    lr,
    weight_decay,
    seed,
):
    return train_linear_classifier(
        z[train_idx],
        labels[train_idx],
        int(labels.max().item()) + 1,
        epochs,
        lr,
        weight_decay,
        seed,
    )


def benchmark_all_languages(
    method,
    x,
    labels,
    language_ids,
    train_idx,
    test_idx,
    id_to_lang,
    args,
    seed,
):
    if method == "inlp":
        z, basis = fit_inlp(
            x,
            language_ids,
            train_idx,
            iters=args.inlp_iters,
            classifier_epochs=args.inlp_classifier_epochs,
            classifier_lr=args.inlp_lr,
            weight_decay=args.weight_decay,
            seed=seed,
        )
        task_head = fit_task_head_on_representation(
            z,
            labels,
            train_idx,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=seed + 17,
        )
        with torch.no_grad():
            task_logits = task_head(z)
        state = {
            "method": "inlp",
            "removed_basis": basis.detach().cpu(),
            "task_head": {k: v.detach().cpu() for k, v in task_head.state_dict().items()},
        }
    else:
        model = fit_projection_method(
            method,
            x,
            labels,
            language_ids,
            train_idx,
            proj_dim=args.proj_dim,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            irm_lambda=args.irm_lambda,
            vrex_lambda=args.vrex_lambda,
            dann_lambda=args.dann_lambda,
            seed=seed,
        )
        model.eval()
        with torch.no_grad():
            z, task_logits = model(x)
        state = {
            "method": method,
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "proj_dim": args.proj_dim,
        }

    per_lang, mean_loss, risk_var, mean_acc = evaluate_task_by_language(
        task_logits, labels, language_ids, test_idx, id_to_lang
    )
    lang_acc = fresh_language_probe_accuracy(
        z,
        language_ids,
        train_idx,
        test_idx,
        len(id_to_lang),
        epochs=args.lang_probe_epochs,
        lr=args.lang_probe_lr,
        weight_decay=args.weight_decay,
        seed=seed + 31,
    )

    return {
        "mean_task_accuracy": mean_acc,
        "mean_task_loss": mean_loss,
        "risk_variance": risk_var,
        "language_probe_accuracy": lang_acc,
        "per_language": per_lang,
        "representation_dim": int(z.shape[1]),
        "state": state,
    }


def leave_one_out_accuracy(
    method,
    x,
    labels,
    language_ids,
    train_mask,
    test_mask,
    heldout_id,
    args,
    seed,
):
    train_idx = torch.where(train_mask & (language_ids != heldout_id))[0]
    test_idx = torch.where(test_mask & (language_ids == heldout_id))[0]

    # Remap the training-language IDs to 0..K-1 for methods that use a
    # language classifier internally (INLP/DANN).
    train_lang_values = sorted(set(language_ids[train_idx].tolist()))
    remap = {old: new for new, old in enumerate(train_lang_values)}
    env_local = language_ids.clone()
    for old, new in remap.items():
        env_local[language_ids == old] = new

    if method == "inlp":
        z, _ = fit_inlp(
            x,
            env_local,
            train_idx,
            iters=args.inlp_iters,
            classifier_epochs=args.inlp_classifier_epochs,
            classifier_lr=args.inlp_lr,
            weight_decay=args.weight_decay,
            seed=seed,
        )
        task_head = fit_task_head_on_representation(
            z,
            labels,
            train_idx,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=seed + 13,
        )
        with torch.no_grad():
            logits = task_head(z[test_idx])
    else:
        model = fit_projection_method(
            method,
            x,
            labels,
            env_local,
            train_idx,
            proj_dim=args.proj_dim,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            irm_lambda=args.irm_lambda,
            vrex_lambda=args.vrex_lambda,
            dann_lambda=args.dann_lambda,
            seed=seed,
        )
        model.eval()
        with torch.no_grad():
            _, logits = model(x[test_idx])

    loss = F.cross_entropy(logits, labels[test_idx]).item()
    acc = accuracy(logits, labels[test_idx])
    return loss, acc, len(test_idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_file", required=True)
    ap.add_argument("--out_dir", default="invariance_analysis/method_benchmark")
    ap.add_argument(
        "--methods",
        nargs="+",
        default=["erm", "irm", "vrex", "dann", "inlp"],
        choices=["erm", "irm", "vrex", "dann", "inlp"],
    )
    ap.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=None,
        help="Default: all layers stored in the features file.",
    )
    ap.add_argument("--proj_dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--irm_lambda", type=float, default=1.0)
    ap.add_argument("--vrex_lambda", type=float, default=10.0)
    ap.add_argument("--dann_lambda", type=float, default=1.0)
    ap.add_argument("--lang_probe_epochs", type=int, default=150)
    ap.add_argument("--lang_probe_lr", type=float, default=1e-2)
    ap.add_argument("--inlp_iters", type=int, default=8)
    ap.add_argument("--inlp_classifier_epochs", type=int, default=100)
    ap.add_argument("--inlp_lr", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_leave_one_out", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    payload = torch.load(args.features_file, map_location="cpu")
    labels = payload["labels"].long().to(device)
    languages = payload["languages"]
    splits = payload["splits"]

    unique_langs = sorted(set(languages))
    lang_to_id = {lang: i for i, lang in enumerate(unique_langs)}
    id_to_lang = {i: lang for lang, i in lang_to_id.items()}
    language_ids = torch.tensor(
        [lang_to_id[x] for x in languages],
        dtype=torch.long,
        device=device,
    )
    train_idx = make_split_indices(splits, "probe_train").to(device)
    test_idx = make_split_indices(splits, "probe_test").to(device)
    train_mask = torch.tensor(
        [s == "probe_train" for s in splits], dtype=torch.bool, device=device
    )
    test_mask = torch.tensor(
        [s == "probe_test" for s in splits], dtype=torch.bool, device=device
    )

    available_layers = sorted(int(k) for k in payload["features"].keys())
    layers = args.layers or available_layers
    missing = sorted(set(layers) - set(available_layers))
    if missing:
        raise ValueError(f"Requested layers not in features file: {missing}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(exist_ok=True)

    summary_rows = []
    loo_rows = []

    for layer in layers:
        x = payload["features"][str(layer)].float().to(device)
        for method_idx, method in enumerate(args.methods):
            seed = args.seed + layer * 1009 + method_idx * 97
            metrics = benchmark_all_languages(
                method,
                x,
                labels,
                language_ids,
                train_idx,
                test_idx,
                id_to_lang,
                args,
                seed,
            )

            row = {
                "layer": layer,
                "method": method,
                "mean_task_accuracy": metrics["mean_task_accuracy"],
                "mean_task_loss": metrics["mean_task_loss"],
                "risk_variance": metrics["risk_variance"],
                "language_probe_accuracy": metrics["language_probe_accuracy"],
                "language_chance_accuracy": 1.0 / len(unique_langs),
                "representation_dim": metrics["representation_dim"],
            }
            for lang, loss, acc, n in metrics["per_language"]:
                row[f"{lang}_task_loss"] = loss
                row[f"{lang}_task_accuracy"] = acc
                row[f"{lang}_n"] = n
            summary_rows.append(row)

            artifact = {
                **metrics["state"],
                "layer": layer,
                "input_dim": int(x.shape[1]),
                "languages": unique_langs,
                "config": vars(args),
            }
            torch.save(artifact, out / "models" / f"{method}_layer_{layer}.pt")

            print(
                f"[all] layer={layer:>2} method={method:<4} "
                f"task={metrics['mean_task_accuracy']:.4f} "
                f"lang={metrics['language_probe_accuracy']:.4f} "
                f"risk_var={metrics['risk_variance']:.6f}"
            )

            if not args.skip_leave_one_out:
                for heldout_id, heldout_lang in id_to_lang.items():
                    loss, acc, n = leave_one_out_accuracy(
                        method,
                        x,
                        labels,
                        language_ids,
                        train_mask,
                        test_mask,
                        heldout_id,
                        args,
                        seed + heldout_id * 53,
                    )
                    loo_rows.append({
                        "layer": layer,
                        "method": method,
                        "heldout_language": heldout_lang,
                        "heldout_task_loss": loss,
                        "heldout_task_accuracy": acc,
                        "n_test": n,
                    })
                    print(
                        f"[loo] layer={layer:>2} method={method:<4} "
                        f"heldout={heldout_lang} acc={acc:.4f}"
                    )

    summary_fields = sorted({k for r in summary_rows for k in r.keys()})
    with (out / "all_language_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        w.writerows(summary_rows)

    if loo_rows:
        with (out / "leave_one_out.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            fields = [
                "layer", "method", "heldout_language",
                "heldout_task_loss", "heldout_task_accuracy", "n_test",
            ]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(loo_rows)

        agg = {}
        for row in loo_rows:
            key = (row["layer"], row["method"])
            agg.setdefault(key, []).append(row["heldout_task_accuracy"])
        aggregate_rows = [
            {
                "layer": layer,
                "method": method,
                "mean_loo_accuracy": float(np.mean(vals)),
                "std_loo_accuracy": float(np.std(vals)),
            }
            for (layer, method), vals in sorted(agg.items())
        ]
        with (out / "leave_one_out_aggregate.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            fields = ["layer", "method", "mean_loo_accuracy", "std_loo_accuracy"]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(aggregate_rows)

    meta = {
        "features_file": args.features_file,
        "methods": args.methods,
        "layers": layers,
        "languages": unique_langs,
        "interpretation": {
            "task_accuracy": "higher is better",
            "language_probe_accuracy": "closer to language chance means more language-oblivious",
            "risk_variance": "lower means more even task risk across languages",
            "mean_loo_accuracy": "higher means better transfer to a language excluded from representation training",
        },
    }
    (out / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved benchmark to {out}")


if __name__ == "__main__":
    main()
