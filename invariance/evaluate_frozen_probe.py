import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class InvariantProbe(nn.Module):
    def __init__(self, input_dim, proj_dim, n_classes):
        super().__init__()
        self.proj = nn.Linear(input_dim, proj_dim, bias=False)
        self.task_head = nn.Linear(proj_dim, n_classes)

    def forward(self, x):
        z = self.proj(x)
        z = F.layer_norm(z, (z.shape[-1],))
        return z, self.task_head(z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe_file", required=True)
    ap.add_argument("--features_files", nargs="+", required=True)
    ap.add_argument("--out_file", required=True)
    args = ap.parse_args()

    probe = torch.load(args.probe_file, map_location="cpu")
    layer = str(probe["layer"])
    model = InvariantProbe(
        probe["input_dim"],
        probe["proj_dim"],
        probe["n_classes"],
    )
    model.load_state_dict(probe["state_dict"])
    model.eval()

    rows = []
    for path in args.features_files:
        payload = torch.load(path, map_location="cpu")
        if layer not in payload["features"]:
            raise ValueError(f"{path} does not contain layer {layer}")

        x = payload["features"][layer].float()
        y = payload["labels"].long()
        langs = payload["languages"]
        splits = payload["splits"]
        test_idx = [i for i, s in enumerate(splits) if s == "probe_test"]

        with torch.no_grad():
            _, logits = model(x[test_idx])

        y_test = y[test_idx]
        lang_test = [langs[i] for i in test_idx]

        file_rows = []
        for lang in sorted(set(lang_test)):
            mask = torch.tensor(
                [item == lang for item in lang_test],
                dtype=torch.bool,
            )
            loss = F.cross_entropy(
                logits[mask],
                y_test[mask],
            ).item()
            acc = (
                logits[mask].argmax(-1) == y_test[mask]
            ).float().mean().item()

            row = {
                "features_file": path,
                "checkpoint": payload.get("checkpoint", ""),
                "layer": int(layer),
                "language": lang,
                "task_loss": loss,
                "task_accuracy": acc,
                "n": int(mask.sum().item()),
            }
            rows.append(row)
            file_rows.append(row)

        mean_acc = np.mean(
            [r["task_accuracy"] for r in file_rows]
        )
        print(
            f"{path}: frozen-probe mean accuracy={mean_acc:.4f}"
        )

    out = Path(args.out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "features_file",
        "checkpoint",
        "layer",
        "language",
        "task_loss",
        "task_accuracy",
        "n",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"Saved {out}")


if __name__ == "__main__":
    main()
