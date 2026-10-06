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


class InvariantProbe(nn.Module):
    def __init__(self, input_dim, proj_dim, n_classes):
        super().__init__()
        self.proj = nn.Linear(input_dim, proj_dim, bias=False)
        self.task_head = nn.Linear(proj_dim, n_classes)

    def forward(self, x):
        z = self.proj(x)
        z = F.layer_norm(z, (z.shape[-1],))
        return z, self.task_head(z)


def irm_penalty(logits, labels):
    scale = torch.tensor(1.0, device=logits.device, requires_grad=True)
    loss = F.cross_entropy(logits * scale, labels)
    grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
    return grad.pow(2)


def make_indices(splits, split_name):
    return torch.tensor(
        [i for i, s in enumerate(splits) if s == split_name],
        dtype=torch.long,
    )


def accuracy(logits, y):
    return (logits.argmax(dim=-1) == y).float().mean().item()


def train_language_probe(
    z_train,
    lang_train,
    z_test,
    lang_test,
    n_lang,
    epochs=150,
    lr=1e-2,
):
    head = nn.Linear(z_train.shape[1], n_lang).to(z_train.device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(head(z_train.detach()), lang_train)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return accuracy(head(z_test), lang_test)


def evaluate_by_language(model, x, y, language_ids, id_to_lang):
    model.eval()
    with torch.no_grad():
        z, logits = model(x)

    rows = []
    losses = []
    accs = []
    for lang_id, lang in id_to_lang.items():
        mask = language_ids == lang_id
        if mask.sum() == 0:
            continue
        loss = F.cross_entropy(logits[mask], y[mask]).item()
        acc = accuracy(logits[mask], y[mask])
        rows.append((lang, loss, acc, int(mask.sum().item())))
        losses.append(loss)
        accs.append(acc)

    return (
        rows,
        float(np.mean(losses)),
        float(np.var(losses)),
        float(np.mean(accs)),
        z,
    )


def fit_one_layer(
    x,
    labels,
    language_ids,
    train_idx,
    test_idx,
    id_to_lang,
    *,
    proj_dim,
    irm_lambda,
    epochs,
    lr,
    weight_decay,
    seed,
    device,
):
    set_seed(seed)
    x = x.float().to(device)
    labels = labels.to(device)
    language_ids = language_ids.to(device)
    train_idx = train_idx.to(device)
    test_idx = test_idx.to(device)

    model = InvariantProbe(
        x.shape[1],
        proj_dim,
        int(labels.max().item()) + 1,
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    train_langs = sorted(set(language_ids[train_idx].tolist()))

    for _ in range(epochs):
        model.train()
        opt.zero_grad(set_to_none=True)

        env_losses = []
        env_penalties = []
        for lang_id in train_langs:
            idx = train_idx[language_ids[train_idx] == lang_id]
            _, logits = model(x[idx])
            env_losses.append(F.cross_entropy(logits, labels[idx]))
            env_penalties.append(irm_penalty(logits, labels[idx]))

        erm = torch.stack(env_losses).mean()
        penalty = torch.stack(env_penalties).mean()
        loss = erm + irm_lambda * penalty
        loss.backward()
        opt.step()

    per_lang, mean_loss, risk_var, mean_acc, _ = evaluate_by_language(
        model,
        x[test_idx],
        labels[test_idx],
        language_ids[test_idx],
        id_to_lang,
    )

    with torch.no_grad():
        z_train, _ = model(x[train_idx])
        z_test, _ = model(x[test_idx])

    lang_acc = train_language_probe(
        z_train,
        language_ids[train_idx],
        z_test,
        language_ids[test_idx],
        len(id_to_lang),
    )

    return model, {
        "mean_task_accuracy": mean_acc,
        "mean_task_loss": mean_loss,
        "risk_variance": risk_var,
        "language_probe_accuracy": lang_acc,
        "per_language": per_lang,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_file", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--proj_dim", type=int, default=64)
    ap.add_argument("--irm_lambda", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--alpha_lang", type=float, default=0.25)
    ap.add_argument("--beta_risk", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )

    payload = torch.load(args.features_file, map_location="cpu")
    labels = payload["labels"].long()
    languages = payload["languages"]
    splits = payload["splits"]

    unique_langs = sorted(set(languages))
    lang_to_id = {lang: i for i, lang in enumerate(unique_langs)}
    id_to_lang = {i: lang for lang, i in lang_to_id.items()}
    language_ids = torch.tensor(
        [lang_to_id[x] for x in languages],
        dtype=torch.long,
    )

    train_idx = make_indices(splits, "probe_train")
    test_idx = make_indices(splits, "probe_test")
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError(
            "features_file must contain probe_train and probe_test examples"
        )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    best = None

    for layer_str, x in payload["features"].items():
        layer = int(layer_str)
        model, metrics = fit_one_layer(
            x,
            labels,
            language_ids,
            train_idx,
            test_idx,
            id_to_lang,
            proj_dim=args.proj_dim,
            irm_lambda=args.irm_lambda,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed + layer,
            device=device,
        )

        chance = 1.0 / len(unique_langs)
        lang_excess = max(
            metrics["language_probe_accuracy"] - chance,
            0.0,
        )
        score = (
            metrics["mean_task_accuracy"]
            - args.alpha_lang * lang_excess
            - args.beta_risk * metrics["risk_variance"]
        )

        row = {
            "layer": layer,
            "mean_task_accuracy": metrics["mean_task_accuracy"],
            "mean_task_loss": metrics["mean_task_loss"],
            "risk_variance": metrics["risk_variance"],
            "language_probe_accuracy": metrics["language_probe_accuracy"],
            "language_chance_accuracy": chance,
            "invariance_score": score,
        }

        for lang, loss, acc, n in metrics["per_language"]:
            row[f"{lang}_task_loss"] = loss
            row[f"{lang}_task_accuracy"] = acc
            row[f"{lang}_n"] = n

        summary_rows.append(row)

        ckpt = {
            "layer": layer,
            "input_dim": x.shape[1],
            "proj_dim": args.proj_dim,
            "n_classes": int(labels.max().item()) + 1,
            "languages": unique_langs,
            "lang_to_id": lang_to_id,
            "state_dict": {
                k: v.detach().cpu()
                for k, v in model.state_dict().items()
            },
            "fit_config": vars(args),
            "metrics": metrics,
            "invariance_score": score,
        }
        torch.save(ckpt, out / f"probe_layer_{layer}.pt")

        if best is None or score > best[0]:
            best = (score, layer)

        print(
            f"layer={layer:>2} "
            f"task_acc={metrics['mean_task_accuracy']:.4f} "
            f"lang_acc={metrics['language_probe_accuracy']:.4f} "
            f"risk_var={metrics['risk_variance']:.6f} "
            f"score={score:.4f}"
        )

    fields = sorted(
        {k for row in summary_rows for k in row.keys()}
    )
    with (out / "layer_summary.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary_rows)

    best_score, best_layer = best
    best_meta = {
        "best_layer": best_layer,
        "best_score": best_score,
        "probe_file": str(out / f"probe_layer_{best_layer}.pt"),
        "features_file": args.features_file,
    }
    (out / "best_probe.json").write_text(
        json.dumps(best_meta, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(best_meta, indent=2))


if __name__ == "__main__":
    main()
