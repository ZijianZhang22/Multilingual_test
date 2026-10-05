import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def auc_loss(g):
    g = g.sort_values("target_tokens_seen")
    g = g.groupby("target_tokens_seen", as_index=False)["val_loss"].mean()
    x = g["target_tokens_seen"].to_numpy(dtype=float)
    y = g["val_loss"].to_numpy(dtype=float)
    if len(x) < 2 or x[-1] == x[0]:
        return np.nan
    return np.trapezoid(y, x) / (x[-1] - x[0])


def tokens_to_threshold(g, threshold):
    g = g.sort_values("target_tokens_seen")
    hit = g[g["val_loss"] <= threshold]
    return np.nan if hit.empty else float(hit.iloc[0]["target_tokens_seen"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_glob", default="runs/seed*/metrics.csv")
    ap.add_argument("--out_dir", default="analysis")
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.metrics_glob))
    if not paths:
        raise FileNotFoundError(f"No files matched {args.metrics_glob}")

    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        parent = Path(p).parent.name
        seed = parent.replace("seed", "") if parent.startswith("seed") else parent
        df["seed"] = seed
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "all_metrics.csv", index=False)

    rows = []
    for (seed, branch), g in df.groupby(["seed", "branch"]):
        ordered = g.sort_values("target_tokens_seen")
        row = {
            "seed": seed,
            "branch": branch,
            "normalized_area_under_loss_curve": auc_loss(g),
            "initial_val_loss": ordered.iloc[0]["val_loss"],
            "final_val_loss": ordered.iloc[-1]["val_loss"],
        }
        if args.threshold is not None:
            row["tokens_to_threshold"] = tokens_to_threshold(g, args.threshold)
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "per_seed_summary.csv", index=False)

    agg_cols = [
        "normalized_area_under_loss_curve",
        "initial_val_loss",
        "final_val_loss",
    ]
    if args.threshold is not None:
        agg_cols.append("tokens_to_threshold")

    agg = summary.groupby("branch")[agg_cols].agg(["mean", "std"])
    agg.to_csv(out / "aggregate_summary.csv")
    print("\nAggregate summary:")
    print(agg)

    curve = (
        df.groupby(["branch", "target_tokens_seen"])["val_loss"]
        .agg(["mean", "std"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    for branch, g in curve.groupby("branch"):
        g = g.sort_values("target_tokens_seen")
        ax.plot(g["target_tokens_seen"], g["mean"], marker="o", label=branch)
        if g["std"].notna().any():
            lo = g["mean"] - g["std"].fillna(0)
            hi = g["mean"] + g["std"].fillna(0)
            ax.fill_between(g["target_tokens_seen"], lo, hi, alpha=0.15)
    ax.set_xlabel("Target-language training tokens")
    ax.set_ylabel("Held-out target validation loss")
    ax.set_title("Future-language adaptation curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "learning_curves.png", dpi=180)
    print(f"\nSaved: {out / 'learning_curves.png'}")


if __name__ == "__main__":
    main()
