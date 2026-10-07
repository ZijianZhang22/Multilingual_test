import argparse
import csv
import subprocess
import sys
from pathlib import Path


def run(cmd):
    cmd = [str(x) for x in cmd]
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def maybe_run(output, cmd, force=False):
    output = Path(output)
    if output.exists() and not force:
        print(f"SKIP existing: {output}")
        return
    run(cmd)


def read_rows(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pick_row(rows, **conds):
    for row in rows:
        if all(str(row.get(k)) == str(v) for k, v in conds.items()):
            return row
    return None


def main():
    ap = argparse.ArgumentParser(
        description=(
            "One-click mechanism analysis on completed RQ1 runs: task-sensitive/null "
            "drift, OLD-vs-NEW gradient interference, layerwise CKA, and principal angles."
        )
    )
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--directions", nargs="+", default=["en:zh", "zh:en"])
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--task_rank", type=int, default=32)
    ap.add_argument("--drift_rank", type=int, default=32)
    ap.add_argument("--cka_layers", nargs="+", type=int, default=[4, 8, 12, 16, 20, 24])
    ap.add_argument("--gradient_layers", nargs="+", type=int, default=[4, 8, 12, 16, 20, 24])
    ap.add_argument("--gradient_batch_size", type=int, default=2)
    ap.add_argument("--gradient_max_batches", type=int, default=8)
    ap.add_argument("--gradient_max_blocks", type=int, default=64)
    ap.add_argument("--extract_batch", type=int, default=16)
    ap.add_argument("--data_dir", default="invariance_data/wiki")
    ap.add_argument("--runs_root", default="invariance_runs")
    ap.add_argument("--rq1_root", default="invariance_analysis/rq1_diagnostics")
    ap.add_argument("--out_root", default="invariance_analysis/mechanism_analysis")
    ap.add_argument(
        "--transferable_subspace_file",
        default="invariance_analysis/causal_representation_suite/transferable_rank64.pt",
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    py = sys.executable
    inv = Path(__file__).resolve().parent
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    aggregate = []

    for seed in args.seeds:
        for direction in args.directions:
            if ":" not in direction:
                raise ValueError(f"Bad direction {direction}; expected OLD:NEW")
            old, new = direction.split(":", 1)
            seq = Path(args.runs_root) / f"sequence_seed{seed}" / f"{old}__{new}"
            anchor_ckpt = seq / f"stage1_{old}"
            post_ckpt = seq / f"stage2_{new}"
            if not anchor_ckpt.exists() or not post_ckpt.exists():
                print(f"WARNING: missing checkpoints for seed={seed} {direction}; skipping")
                continue

            rq1 = Path(args.rq1_root) / f"seed{seed}" / f"{old}_to_{new}"
            data_file = rq1 / "xnli_probe.jsonl"
            anchor_l12 = rq1 / "anchor_features.pt"
            post_l12 = rq1 / "post_features.pt"
            lang_subspace = rq1 / "anchor_inlp_language_subspace.pt"
            if not all(p.exists() for p in [data_file, anchor_l12, post_l12, lang_subspace]):
                raise FileNotFoundError(
                    f"RQ1 artifacts missing for seed={seed} {direction}. "
                    f"Run run_rq1_forgetting_diagnostics.py first."
                )

            root = out_root / f"seed{seed}" / f"{old}_to_{new}"
            root.mkdir(parents=True, exist_ok=True)

            task_drift = root / "task_visible_null_drift.csv"
            task_subspace = root / "task_sensitive_subspace.pt"
            maybe_run(
                task_drift,
                [
                    py, inv / "analyze_task_visible_drift.py",
                    "--anchor_checkpoint", anchor_ckpt,
                    "--anchor_features", anchor_l12,
                    "--post_features", post_l12,
                    "--data_dir", args.data_dir,
                    "--old_language", old,
                    "--layer", args.layer,
                    "--rank", args.task_rank,
                    "--gradient_batch_size", args.gradient_batch_size,
                    "--gradient_max_batches", args.gradient_max_batches,
                    "--max_blocks", args.gradient_max_blocks,
                    "--seed", seed,
                    "--out_file", task_drift,
                    "--subspace_out", task_subspace,
                ],
                force=args.force,
            )

            grad_file = root / "gradient_interference.csv"
            maybe_run(
                grad_file,
                [
                    py, inv / "analyze_gradient_interference.py",
                    "--checkpoint", anchor_ckpt,
                    "--data_dir", args.data_dir,
                    "--old_language", old,
                    "--new_language", new,
                    "--layers", *args.gradient_layers,
                    "--batch_size", args.gradient_batch_size,
                    "--max_batches", args.gradient_max_batches,
                    "--max_blocks", args.gradient_max_blocks,
                    "--seed", seed,
                    "--out_file", grad_file,
                ],
                force=args.force,
            )

            anchor_multi = root / "anchor_multilayer_features.pt"
            post_multi = root / "post_multilayer_features.pt"
            maybe_run(
                anchor_multi,
                [
                    py, inv / "extract_hidden.py",
                    "--checkpoint", anchor_ckpt,
                    "--data_file", data_file,
                    "--out_file", anchor_multi,
                    "--layers", *args.cka_layers,
                    "--batch_size", args.extract_batch,
                ],
                force=args.force,
            )
            maybe_run(
                post_multi,
                [
                    py, inv / "extract_hidden.py",
                    "--checkpoint", post_ckpt,
                    "--data_file", data_file,
                    "--out_file", post_multi,
                    "--layers", *args.cka_layers,
                    "--batch_size", args.extract_batch,
                ],
                force=args.force,
            )

            cka_file = root / "layerwise_cka.csv"
            maybe_run(
                cka_file,
                [
                    py, inv / "analyze_layerwise_cka.py",
                    "--anchor_features", anchor_multi,
                    "--post_features", post_multi,
                    "--layers", *args.cka_layers,
                    "--out_file", cka_file,
                ],
                force=args.force,
            )

            angles_file = root / "subspace_angles.csv"
            cmd = [
                py, inv / "analyze_subspace_angles.py",
                "--language_subspace_file", lang_subspace,
                "--task_subspace_file", task_subspace,
                "--anchor_features", anchor_l12,
                "--post_features", post_l12,
                "--old_language", old,
                "--layer", args.layer,
                "--drift_rank", args.drift_rank,
                "--drift_subspace_out", root / "drift_subspace.pt",
                "--out_file", angles_file,
            ]
            transfer = Path(args.transferable_subspace_file)
            if transfer.exists():
                cmd += ["--transferable_subspace_file", transfer]
            else:
                print(f"NOTE: transferable subspace not found at {transfer}; angles will omit it")
            maybe_run(angles_file, cmd, force=args.force)

            task_rows = read_rows(task_drift)
            grad_rows = read_rows(grad_file)
            cka_rows = read_rows(cka_file)
            angle_rows = read_rows(angles_file)

            task_old = pick_row(task_rows, language=old)
            grad_l = pick_row(grad_rows, layer=args.layer)
            cka_old = pick_row(cka_rows, layer=args.layer, language=old)
            drift_task = None
            drift_lang = None
            for r in angle_rows:
                pair = {r["subspace_a"], r["subspace_b"]}
                if pair == {"drift", "task_sensitive"}:
                    drift_task = r
                if pair == {"drift", "language"}:
                    drift_lang = r

            rec = {
                "seed": seed,
                "direction": direction,
                "old_language": old,
                "new_language": new,
            }
            if task_old:
                for key in [
                    "total_drift_l2",
                    "task_sensitive_drift_l2",
                    "task_null_drift_l2",
                    "task_sensitive_drift_per_dim",
                    "task_null_drift_per_dim",
                ]:
                    rec[key] = task_old[key]
            if grad_l:
                rec["gradient_cosine_layer"] = grad_l["mean_gradient_cosine"]
                rec["gradient_negative_batch_fraction"] = grad_l["negative_batch_fraction"]
            if cka_old:
                rec["old_language_cka_layer"] = cka_old["linear_cka"]
                rec["old_language_relative_l2_layer"] = cka_old["relative_l2_drift"]
            if drift_task:
                rec["drift_task_mean_angle_deg"] = drift_task["principal_angle_mean_deg"]
                rec["drift_task_overlap"] = drift_task["normalized_projection_overlap"]
            if drift_lang:
                rec["drift_language_mean_angle_deg"] = drift_lang["principal_angle_mean_deg"]
                rec["drift_language_overlap"] = drift_lang["normalized_projection_overlap"]
            aggregate.append(rec)

    if aggregate:
        agg_file = out_root / "aggregate_mechanism_summary.csv"
        fields = sorted({k for row in aggregate for k in row})
        with agg_file.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(aggregate)
        print(f"\nSaved aggregate mechanism summary: {agg_file}")

    print("\n=== EXTENDED MECHANISM ANALYSIS COMPLETE ===")
    print(f"Outputs: {out_root}")


if __name__ == "__main__":
    main()
