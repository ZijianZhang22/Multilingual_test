import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd


def parse_sequence_stage(checkpoint):
    p = Path(str(checkpoint))
    stage_name = p.name
    sequence = p.parent.name
    if not stage_name.startswith("stage"):
        raise ValueError(f"Cannot parse stage from checkpoint: {checkpoint}")
    prefix = stage_name.split("_", 1)[0]
    stage = int(prefix.replace("stage", ""))
    return sequence, stage


def pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, len(x)
    return float(np.corrcoef(x, y)[0, 1]), len(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence_metrics", required=True)
    ap.add_argument("--invariance_summary", required=True)
    ap.add_argument("--out_dir", default="invariance_analysis/forgetting_link")
    args = ap.parse_args()

    seq = pd.read_csv(args.sequence_metrics)
    inv = pd.read_csv(args.invariance_summary)

    parsed = inv["checkpoint"].apply(parse_sequence_stage)
    inv["sequence"] = [x[0] for x in parsed]
    inv["stage"] = [x[1] for x in parsed]

    rows = []
    for sequence, g_inv in inv.groupby("sequence"):
        base_inv = g_inv[g_inv["stage"] == 0]
        if base_inv.empty:
            print(f"WARNING: no stage0 invariant row for sequence={sequence}; skipping")
            continue

        base_frozen = float(base_inv.iloc[0]["frozen_mean_accuracy"])
        base_refit = float(base_inv.iloc[0]["refit_mean_accuracy"])

        g_seq = seq[seq["sequence"] == sequence]
        base_loss = (
            g_seq[g_seq["stage"] == 0]
            .set_index("eval_language")["val_loss"]
            .to_dict()
        )

        for _, ir in g_inv.iterrows():
            stage = int(ir["stage"])
            if stage == 0:
                continue

            current = g_seq[g_seq["stage"] == stage]
            for _, sr in current.iterrows():
                lang = sr["eval_language"]
                if lang not in base_loss:
                    continue

                forgetting = float(sr["val_loss"]) - float(base_loss[lang])
                inv_drop = base_frozen - float(ir["frozen_mean_accuracy"])
                refit_drop = base_refit - float(ir["refit_mean_accuracy"])

                rows.append({
                    "sequence": sequence,
                    "stage": stage,
                    "eval_language": lang,
                    "val_loss": float(sr["val_loss"]),
                    "base_val_loss": float(base_loss[lang]),
                    "forgetting_loss_delta": forgetting,
                    "frozen_mean_accuracy": float(ir["frozen_mean_accuracy"]),
                    "refit_mean_accuracy": float(ir["refit_mean_accuracy"]),
                    "invariant_access_drop": inv_drop,
                    "refit_information_drop": refit_drop,
                    "recovery_gap_accuracy": float(ir["recovery_gap_accuracy"]),
                })

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joined = pd.DataFrame(rows)
    joined.to_csv(out / "joined_forgetting_invariance.csv", index=False)

    summaries = []
    for metric in [
        "invariant_access_drop",
        "refit_information_drop",
        "recovery_gap_accuracy",
    ]:
        r, n = pearson(joined[metric], joined["forgetting_loss_delta"])
        summaries.append({
            "scope": "all",
            "metric": metric,
            "pearson_r_with_forgetting": r,
            "n": n,
        })

        for lang, g in joined.groupby("eval_language"):
            r_lang, n_lang = pearson(g[metric], g["forgetting_loss_delta"])
            summaries.append({
                "scope": f"language:{lang}",
                "metric": metric,
                "pearson_r_with_forgetting": r_lang,
                "n": n_lang,
            })

    summary = pd.DataFrame(summaries)
    summary.to_csv(out / "correlations.csv", index=False)

    print("\nCorrelation with old-language loss increase:")
    for _, row in summary.iterrows():
        print(
            f"{row['scope']:>14} | {row['metric']:<24} "
            f"r={row['pearson_r_with_forgetting']!s:<8} n={int(row['n'])}"
        )

    print(
        "\nInterpretation: invariant_access_drop uses the frozen base probe; "
        "refit_information_drop asks whether a newly fitted probe also loses task information. "
        "With only two language orders and one seed, correlations are diagnostic only."
    )
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
