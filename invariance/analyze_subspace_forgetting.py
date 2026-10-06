import argparse
from pathlib import Path

import numpy as np
import pandas as pd


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
    ap.add_argument("--subspace_by_language", required=True)
    ap.add_argument("--out_dir", default="invariance_analysis/subspace_forgetting")
    args = ap.parse_args()

    seq = pd.read_csv(args.sequence_metrics)
    sub = pd.read_csv(args.subspace_by_language)

    rows = []
    for sequence, g in seq.groupby("sequence"):
        g = g.sort_values(["stage", "eval_language"])
        sub_g = sub[sub["sequence"] == sequence]

        # For each language, forgetting is measured from its most recent
        # "anchor" checkpoint: the latest stage at which that language was
        # explicitly trained. Before first exposure, the anchor is stage 0.
        anchors = {}
        stage0 = g[g["stage"] == 0]
        for _, r in stage0.iterrows():
            anchors[r["eval_language"]] = {
                "stage": 0,
                "loss": float(r["val_loss"]),
            }

        for stage in sorted(s for s in g["stage"].unique() if s > 0):
            cur = g[g["stage"] == stage]
            if cur.empty:
                continue
            trained_language = str(cur.iloc[0]["trained_language"])

            # Evaluate forgetting BEFORE resetting the trained language's
            # anchor to its new post-training state.
            for _, r in cur.iterrows():
                lang = str(r["eval_language"])
                if lang not in anchors:
                    continue

                anchor = anchors[lang]
                current_loss = float(r["val_loss"])
                forgetting = current_loss - float(anchor["loss"])
                was_old_language = (
                    lang != trained_language and int(anchor["stage"]) > 0
                )

                sr = sub_g[
                    (sub_g["stage"] == stage) &
                    (sub_g["language"] == lang)
                ]
                if sr.empty:
                    continue
                srow = sr.iloc[0].to_dict()

                rows.append({
                    "sequence": sequence,
                    "stage": int(stage),
                    "trained_language": trained_language,
                    "eval_language": lang,
                    "anchor_stage": int(anchor["stage"]),
                    "anchor_val_loss": float(anchor["loss"]),
                    "current_val_loss": current_loss,
                    "forgetting_loss_delta": forgetting,
                    "is_old_language": bool(was_old_language),
                    **{
                        k: v
                        for k, v in srow.items()
                        if k not in {
                            "sequence", "stage", "trained_language",
                            "language", "features_file", "checkpoint",
                        }
                    },
                })

            # Once a language is trained at this stage, its post-training
            # performance becomes the anchor for future forgetting.
            trained_rows = cur[cur["eval_language"] == trained_language]
            if not trained_rows.empty:
                anchors[trained_language] = {
                    "stage": int(stage),
                    "loss": float(trained_rows.iloc[0]["val_loss"]),
                }

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    joined = pd.DataFrame(rows)
    joined.to_csv(out / "joined_subspace_forgetting.csv", index=False)

    old = joined[joined["is_old_language"] == True].copy()
    old.to_csv(out / "old_language_forgetting.csv", index=False)

    metrics = [
        "mean_total_drift_l2",
        "mean_lang_drift_l2",
        "mean_shared_drift_l2",
        "lang_drift_fraction",
        "shared_drift_fraction",
        "lang_drift_per_dim",
        "shared_drift_per_dim",
        "lang_vs_shared_drift_per_dim_ratio",
        "lang_subspace_overlap",
        "mean_principal_angle_deg",
        "max_principal_angle_deg",
        "lang_representation_energy_fraction",
    ]

    summaries = []
    for scope_name, frame in [("all", joined), ("old_languages", old)]:
        for metric in metrics:
            if metric not in frame.columns:
                continue
            r, n = pearson(frame[metric], frame["forgetting_loss_delta"])
            summaries.append({
                "scope": scope_name,
                "metric": metric,
                "pearson_r_with_forgetting": r,
                "n": n,
            })

    corr = pd.DataFrame(summaries)
    corr.to_csv(out / "correlations.csv", index=False)

    print("\nOld-language forgetting rows:")
    if old.empty:
        print(
            "None yet. With only two-stage orders there is one old-language "
            "forgetting observation per branch; use more orders/seeds for real statistics."
        )
    else:
        cols = [
            "sequence", "stage", "trained_language", "eval_language",
            "forgetting_loss_delta", "mean_lang_drift_l2",
            "mean_shared_drift_l2", "lang_subspace_overlap",
        ]
        print(old[cols].to_string(index=False))

    print("\nDiagnostic correlations:")
    for _, r in corr.iterrows():
        print(
            f"{r['scope']:>13} | {r['metric']:<38} "
            f"r={r['pearson_r_with_forgetting']} n={int(r['n'])}"
        )

    print(
        "\nImportant: correlations from two language orders / one seed are "
        "diagnostic only. The meaningful experiment needs multiple seeds and "
        "more language orders."
    )
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
