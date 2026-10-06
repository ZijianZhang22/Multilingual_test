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


def fit_probe(
    x,
    y,
    language_ids,
    train_idx,
    *,
    proj_dim,
    n_classes,
    irm_lambda,
    epochs,
    lr,
    weight_decay,
    seed,
    device,
):
    set_seed(seed)
    model = InvariantProbe(x.shape[1], proj_dim, n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    x = x.to(device)
    y = y.to(device)
    language_ids = language_ids.to(device)
    train_idx = train_idx.to(device)
    train_lang_ids = sorted(set(language_ids[train_idx].tolist()))

    for _ in range(epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        env_losses = []
        env_penalties = []
        for lang_id in train_lang_ids:
            idx = train_idx[language_ids[train_idx] == lang_id]
            _, logits = model(x[idx])
            env_losses.append(F.cross_entropy(logits, y[idx]))
            if irm_lambda > 0:
                env_penalties.append(irm_penalty(logits, y[idx]))

        erm = torch.stack(env_losses).mean()
        penalty = torch.stack(env_penalties).mean() if env_penalties else 0.0
        loss = erm + irm_lambda * penalty
        loss.backward()
        opt.step()

    return model


def eval_probe(model, x, y, langs, test_idx):
    model.eval()
    with torch.no_grad():
        _, logits = model(x[test_idx])
    y_test = y[test_idx]
    lang_test = [langs[i] for i in test_idx.tolist()]

    rows = []
    for lang in sorted(set(lang_test)):
        mask = torch.tensor([v == lang for v in lang_test], device=x.device)
        loss = F.cross_entropy(logits[mask], y_test[mask]).item()
        acc = (logits[mask].argmax(-1) == y_test[mask]).float().mean().item()
        rows.append((lang, loss, acc, int(mask.sum().item())))
    mean_acc = float(np.mean([r[2] for r in rows]))
    mean_loss = float(np.mean([r[1] for r in rows]))
    return rows, mean_acc, mean_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference_probe", required=True)
    ap.add_argument("--features_files", nargs="+", required=True)
    ap.add_argument("--out_file", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight_decay", type=float, default=None)
    ap.add_argument("--irm_lambda", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )

    probe_ckpt = torch.load(args.reference_probe, map_location="cpu")
    layer = str(probe_ckpt["layer"])
    fit_cfg = probe_ckpt.get("fit_config", {})
    epochs = args.epochs if args.epochs is not None else int(fit_cfg.get("epochs", 250))
    lr = args.lr if args.lr is not None else float(fit_cfg.get("lr", 1e-3))
    weight_decay = (
        args.weight_decay
        if args.weight_decay is not None
        else float(fit_cfg.get("weight_decay", 1e-4))
    )
    irm_lambda = (
        args.irm_lambda
        if args.irm_lambda is not None
        else float(fit_cfg.get("irm_lambda", 1.0))
    )

    frozen = InvariantProbe(
        probe_ckpt["input_dim"],
        probe_ckpt["proj_dim"],
        probe_ckpt["n_classes"],
    ).to(device)
    frozen.load_state_dict(probe_ckpt["state_dict"])
    frozen.eval()

    all_rows = []
    summary_rows = []

    for file_idx, path in enumerate(args.features_files):
        payload = torch.load(path, map_location="cpu")
        if layer not in payload["features"]:
            raise ValueError(f"{path} does not contain layer {layer}")

        x = payload["features"][layer].float().to(device)
        y = payload["labels"].long().to(device)
        langs = payload["languages"]
        splits = payload["splits"]
        unique_langs = sorted(set(langs))
        lang_to_id = {lang: i for i, lang in enumerate(unique_langs)}
        language_ids = torch.tensor(
            [lang_to_id[v] for v in langs], dtype=torch.long, device=device
        )
        train_idx = torch.tensor(
            [i for i, s in enumerate(splits) if s == "probe_train"],
            dtype=torch.long,
            device=device,
        )
        test_idx = torch.tensor(
            [i for i, s in enumerate(splits) if s == "probe_test"],
            dtype=torch.long,
            device=device,
        )

        frozen_by_lang, frozen_mean_acc, frozen_mean_loss = eval_probe(
            frozen, x, y, langs, test_idx
        )

        refit = fit_probe(
            x,
            y,
            language_ids,
            train_idx,
            proj_dim=probe_ckpt["proj_dim"],
            n_classes=probe_ckpt["n_classes"],
            irm_lambda=irm_lambda,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            seed=args.seed + file_idx * 1009,
            device=device,
        )
        refit_by_lang, refit_mean_acc, refit_mean_loss = eval_probe(
            refit, x, y, langs, test_idx
        )
        refit_map = {lang: (loss, acc, n) for lang, loss, acc, n in refit_by_lang}

        checkpoint = payload.get("checkpoint", "")
        for lang, frozen_loss, frozen_acc, n in frozen_by_lang:
            refit_loss, refit_acc, _ = refit_map[lang]
            all_rows.append({
                "features_file": path,
                "checkpoint": checkpoint,
                "layer": int(layer),
                "language": lang,
                "frozen_task_loss": frozen_loss,
                "frozen_task_accuracy": frozen_acc,
                "refit_task_loss": refit_loss,
                "refit_task_accuracy": refit_acc,
                "refit_minus_frozen_accuracy": refit_acc - frozen_acc,
                "n": n,
            })

        summary_rows.append({
            "features_file": path,
            "checkpoint": checkpoint,
            "layer": int(layer),
            "frozen_mean_accuracy": frozen_mean_acc,
            "frozen_mean_loss": frozen_mean_loss,
            "refit_mean_accuracy": refit_mean_acc,
            "refit_mean_loss": refit_mean_loss,
            "recovery_gap_accuracy": refit_mean_acc - frozen_mean_acc,
        })

        print(
            f"{path}: frozen_acc={frozen_mean_acc:.4f} "
            f"refit_acc={refit_mean_acc:.4f} "
            f"recovery_gap={refit_mean_acc - frozen_mean_acc:+.4f}"
        )

    out = Path(args.out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "features_file", "checkpoint", "layer", "language",
        "frozen_task_loss", "frozen_task_accuracy",
        "refit_task_loss", "refit_task_accuracy",
        "refit_minus_frozen_accuracy", "n",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)

    summary_path = out.with_name(out.stem + "_summary.csv")
    fields2 = [
        "features_file", "checkpoint", "layer",
        "frozen_mean_accuracy", "frozen_mean_loss",
        "refit_mean_accuracy", "refit_mean_loss",
        "recovery_gap_accuracy",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields2)
        w.writeheader()
        w.writerows(summary_rows)

    print(f"Saved: {out}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
