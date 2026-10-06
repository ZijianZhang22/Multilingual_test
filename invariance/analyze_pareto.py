import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def pareto_mask(df):
    # Lower forgetting is better; higher new-language gain is better.
    pts = df[["forgetting_loss_delta", "new_language_gain"]].to_numpy(float)
    keep = np.ones(len(df), dtype=bool)
    for i, (f_i, g_i) in enumerate(pts):
        dominated = False
        for j, (f_j, g_j) in enumerate(pts):
            if i == j:
                continue
            if (f_j <= f_i and g_j >= g_i) and (f_j < f_i or g_j > g_i):
                dominated = True
                break
        keep[i] = not dominated
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out_dir", default="invariance_analysis/pareto")
    args = ap.parse_args()

    frames = []
    for path in args.inputs:
        df = pd.read_csv(path)
        df["source_file"] = path
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)

    group_cols = ["method", "lambda"]
    agg = (
        all_df.groupby(group_cols, dropna=False)
        .agg(
            n=("seed", "count"),
            forgetting_mean=("forgetting_loss_delta", "mean"),
            forgetting_std=("forgetting_loss_delta", "std"),
            new_gain_mean=("new_language_gain", "mean"),
            new_gain_std=("new_language_gain", "std"),
        )
        .reset_index()
    )

    proxy = agg.rename(columns={
        "forgetting_mean": "forgetting_loss_delta",
        "new_gain_mean": "new_language_gain",
    })
    agg["pareto"] = pareto_mask(proxy)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(out / "all_interventions.csv", index=False)
    agg.to_csv(out / "aggregate_pareto.csv", index=False)
    agg[agg["pareto"]].to_csv(out / "pareto_front.csv", index=False)

    print("\n=== Pareto front (mean across supplied runs) ===")
    show = agg[agg["pareto"]].sort_values(
        ["forgetting_mean", "new_gain_mean"], ascending=[True, False]
    )
    print(show.to_string(index=False))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
