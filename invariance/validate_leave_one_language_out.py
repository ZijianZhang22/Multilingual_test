import argparse
import csv
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


def accuracy(logits, labels):
    return (logits.argmax(dim=-1) == labels).float().mean().item()


def fit_probe(
    x,
    y,
    language_ids,
    train_idx,
    train_lang_ids,
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
    model = InvariantProbe(
        x.shape[1],
        proj_dim,
        int(y.max().item()) + 1,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    x = x.to(device)
    y = y.to(device)
    language_ids = language_ids.to(device)
    train_idx = train_idx.to(device)

    for _ in range(epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        env_losses = []
        penalties = []
        for lang_id in train_lang_ids:
            idx = train_idx[language_ids[train_idx] == lang_id]
            if len(idx) == 0:
                continue
            _, logits = model(x[idx])
            env_losses.append(F.cross_entropy(logits, y[idx]))
            if irm_lambda > 0:
                penalties.append(irm_penalty(logits, y[idx]))

        erm = torch.stack(env_losses).mean()
        if penalties:
            penalty = torch.stack(penalties).mean()
            loss = erm + irm_lambda * penalty
        else:
            loss = erm
        loss.backward()
        opt.step()

    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_file", required=True)
    ap.add_argument("--out_file", required=True)
    ap.add_argument("--proj_dim", type=int, default=64)
    ap.add_argument("--irm_lambdas", type=float, nargs="+", default=[0.0, 1.0])
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
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
    language_ids = torch.tensor([lang_to_id[x] for x in languages], dtype=torch.long)

    train_mask = torch.tensor([s == "probe_train" for s in splits], dtype=torch.bool)
    test_mask = torch.tensor([s == "probe_test" for s in splits], dtype=torch.bool)

    rows = []
    for layer_str, features in payload["features"].items():
        layer = int(layer_str)
        x = features.float()

        for heldout in unique_langs:
            heldout_id = lang_to_id[heldout]
            train_langs = [l for l in unique_langs if l != heldout]
            train_lang_ids = [lang_to_id[l] for l in train_langs]

            train_idx = torch.where(
                train_mask & (language_ids != heldout_id)
            )[0]
            test_idx = torch.where(
                test_mask & (language_ids == heldout_id)
            )[0]

            if len(train_idx) == 0 or len(test_idx) == 0:
                raise ValueError(
                    f"Missing train/test examples for held-out language {heldout}"
                )

            for irm_lambda in args.irm_lambdas:
                model = fit_probe(
                    x,
                    labels,
                    language_ids,
                    train_idx,
                    train_lang_ids,
                    proj_dim=args.proj_dim,
                    irm_lambda=irm_lambda,
                    epochs=args.epochs,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    seed=args.seed + layer * 101 + heldout_id,
                    device=device,
                )

                model.eval()
                with torch.no_grad():
                    _, logits = model(x[test_idx].to(device))
                y_test = labels[test_idx].to(device)
                loss = F.cross_entropy(logits, y_test).item()
                acc = accuracy(logits, y_test)

                row = {
                    "layer": layer,
                    "heldout_language": heldout,
                    "train_languages": ",".join(train_langs),
                    "irm_lambda": irm_lambda,
                    "method": "IRM" if irm_lambda > 0 else "ERM",
                    "heldout_task_loss": loss,
                    "heldout_task_accuracy": acc,
                    "n_test": len(test_idx),
                }
                rows.append(row)
                print(
                    f"layer={layer:>2} heldout={heldout} "
                    f"method={row['method']} acc={acc:.4f} loss={loss:.4f}"
                )

    out = Path(args.out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "layer",
        "heldout_language",
        "train_languages",
        "irm_lambda",
        "method",
        "heldout_task_loss",
        "heldout_task_accuracy",
        "n_test",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print("\nMean leave-one-language-out accuracy:")
    for layer in sorted({r["layer"] for r in rows}):
        for method in sorted({r["method"] for r in rows}):
            vals = [
                r["heldout_task_accuracy"]
                for r in rows
                if r["layer"] == layer and r["method"] == method
            ]
            if vals:
                print(f"layer={layer:>2} method={method}: {np.mean(vals):.4f}")

    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
