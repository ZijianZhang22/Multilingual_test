import argparse
import csv
import math
from pathlib import Path

import torch

from benchmark_representation_methods import fit_inlp, make_split_indices


def parse_sequence_stage(checkpoint):
    p = Path(str(checkpoint))
    stage_name = p.name
    sequence = p.parent.name
    if not stage_name.startswith("stage"):
        return "", -1, stage_name
    prefix = stage_name.split("_", 1)[0]
    try:
        stage = int(prefix.replace("stage", ""))
    except ValueError:
        stage = -1
    trained = stage_name.split("_", 1)[1] if "_" in stage_name else ""
    return sequence, stage, trained


def ensure_aligned(reference, current, path):
    for key in ["example_ids", "languages", "splits"]:
        if reference.get(key) != current.get(key):
            raise ValueError(
                f"{path}: {key} does not align with the reference features. "
                "Extract every checkpoint on the same XNLI JSONL in the same order."
            )
    if not torch.equal(reference["labels"], current["labels"]):
        raise ValueError(f"{path}: labels do not align with reference features")


def squared_norm_rows(x):
    return (x.float() * x.float()).sum(dim=1)


def drift_metrics(delta, q_lang):
    total_e = squared_norm_rows(delta)
    lang_coords = delta @ q_lang
    lang_e = squared_norm_rows(lang_coords)
    shared_e = (total_e - lang_e).clamp_min(0.0)

    d = delta.shape[1]
    k = q_lang.shape[1]
    shared_dim = max(d - k, 1)

    total_sum = float(total_e.sum().item())
    lang_sum = float(lang_e.sum().item())
    shared_sum = float(shared_e.sum().item())

    total_per_dim = float(total_e.mean().item()) / d
    lang_per_dim = float(lang_e.mean().item()) / max(k, 1)
    shared_per_dim = float(shared_e.mean().item()) / shared_dim

    return {
        "mean_total_drift_l2": math.sqrt(float(total_e.mean().item())),
        "mean_lang_drift_l2": math.sqrt(float(lang_e.mean().item())),
        "mean_shared_drift_l2": math.sqrt(float(shared_e.mean().item())),
        "lang_drift_fraction": lang_sum / total_sum if total_sum > 0 else 0.0,
        "shared_drift_fraction": shared_sum / total_sum if total_sum > 0 else 0.0,
        "lang_drift_per_dim": lang_per_dim,
        "shared_drift_per_dim": shared_per_dim,
        "lang_vs_shared_drift_per_dim_ratio": (
            lang_per_dim / shared_per_dim if shared_per_dim > 0 else float("nan")
        ),
        "total_drift_per_dim": total_per_dim,
    }


def prefix_dict(d, prefix):
    return {f"{prefix}{k}": v for k, v in d.items()}


def representation_energy_metrics(x, q_lang):
    total_e = squared_norm_rows(x)
    lang_e = squared_norm_rows(x @ q_lang)
    total = float(total_e.sum().item())
    lang = float(lang_e.sum().item())
    return {
        "lang_representation_energy_fraction": lang / total if total > 0 else 0.0,
    }


def subspace_similarity(q0, qt):
    if q0.shape[1] == 0 or qt.shape[1] == 0:
        return {
            "refit_lang_rank": int(qt.shape[1]),
            "lang_subspace_overlap": float("nan"),
            "mean_principal_angle_deg": float("nan"),
            "max_principal_angle_deg": float("nan"),
        }

    s = torch.linalg.svdvals(q0.T @ qt).float().clamp(0.0, 1.0)
    angles = torch.rad2deg(torch.acos(s))
    denom = min(q0.shape[1], qt.shape[1])
    overlap = float((s * s).sum().item() / max(denom, 1))
    return {
        "refit_lang_rank": int(qt.shape[1]),
        "lang_subspace_overlap": overlap,
        "mean_principal_angle_deg": float(angles.mean().item()),
        "max_principal_angle_deg": float(angles.max().item()),
    }


def find_anchor_stage(records_for_sequence, current_stage, language):
    """Latest PRIOR stage at which language was trained; otherwise stage 0."""
    candidates = [
        stage
        for stage, rec in records_for_sequence.items()
        if stage < current_stage and rec["trained_language"] == language
    ]
    return max(candidates) if candidates else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference_features", required=True)
    ap.add_argument("--features_files", nargs="+", required=True)
    ap.add_argument("--out_dir", default="invariance_analysis/subspace_tracking")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--inlp_iters", type=int, default=32)
    ap.add_argument("--inlp_classifier_epochs", type=int, default=100)
    ap.add_argument("--inlp_lr", type=float, default=1e-2)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    ref = torch.load(args.reference_features, map_location="cpu")
    layer = str(args.layer)
    if layer not in ref["features"]:
        raise ValueError(f"Layer {args.layer} not found in reference features")

    x0 = ref["features"][layer].float().to(device)
    languages = ref["languages"]
    splits = ref["splits"]
    unique_langs = sorted(set(languages))
    lang_to_id = {lang: i for i, lang in enumerate(unique_langs)}
    language_ids = torch.tensor(
        [lang_to_id[v] for v in languages], dtype=torch.long, device=device
    )
    train_idx = make_split_indices(splits, "probe_train").to(device)

    print(
        f"Fitting reference INLP-{args.inlp_iters} language subspace "
        f"at layer {args.layer}..."
    )
    _, q_ref = fit_inlp(
        x0,
        language_ids,
        train_idx,
        iters=args.inlp_iters,
        classifier_epochs=args.inlp_classifier_epochs,
        classifier_lr=args.inlp_lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    q_ref = q_ref.to(device)
    print(
        f"Reference language-subspace rank: {q_ref.shape[1]} / {x0.shape[1]}"
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "layer": args.layer,
            "inlp_iters": args.inlp_iters,
            "input_dim": x0.shape[1],
            "language_subspace_basis": q_ref.detach().cpu(),
            "languages": unique_langs,
            "reference_features": args.reference_features,
        },
        out / "reference_inlp_language_subspace.pt",
    )

    # Load every checkpoint first so we can compute drift from the correct
    # pre-forgetting anchor, not only from the pretrained base.
    records = {}
    for path in args.features_files:
        cur = torch.load(path, map_location="cpu")
        ensure_aligned(ref, cur, path)
        if layer not in cur["features"]:
            raise ValueError(f"{path}: layer {args.layer} not found")

        checkpoint = cur.get("checkpoint", "")
        sequence, stage, trained_language = parse_sequence_stage(checkpoint)
        if not sequence or stage < 0:
            raise ValueError(f"Could not parse sequence/stage from {checkpoint}")

        rec = {
            "path": path,
            "checkpoint": checkpoint,
            "sequence": sequence,
            "stage": stage,
            "trained_language": trained_language,
            "x": cur["features"][layer].float().to(device),
        }
        records.setdefault(sequence, {})[stage] = rec

    # Fit checkpoint-specific language subspaces with THE SAME seed at every
    # checkpoint. Otherwise identical stage-0 features can appear to have a
    # rotated subspace purely because INLP classifiers were initialized
    # differently.
    for sequence, seq_records in records.items():
        for stage, rec in sorted(seq_records.items()):
            _, q_t = fit_inlp(
                rec["x"],
                language_ids,
                train_idx,
                iters=args.inlp_iters,
                classifier_epochs=args.inlp_classifier_epochs,
                classifier_lr=args.inlp_lr,
                weight_decay=args.weight_decay,
                seed=args.seed,
            )
            rec["q"] = q_t.to(device)

    summary_rows = []
    lang_rows = []

    for sequence, seq_records in records.items():
        if 0 not in seq_records:
            raise ValueError(f"{sequence}: missing stage0 checkpoint")

        for stage, rec in sorted(seq_records.items()):
            xt = rec["x"]
            q_t = rec["q"]
            checkpoint = rec["checkpoint"]
            trained_language = rec["trained_language"]

            base_delta = xt - x0
            base_drift = drift_metrics(base_delta, q_ref)
            global_energy = representation_energy_metrics(xt, q_ref)
            base_subspace = subspace_similarity(q_ref, q_t)

            # Also track one-step drift for a compact global diagnostic.
            prev_stage = max([s for s in seq_records if s < stage], default=0)
            prev_rec = seq_records[prev_stage]
            step_delta = xt - prev_rec["x"]
            step_drift = drift_metrics(step_delta, q_ref)
            step_subspace = subspace_similarity(prev_rec["q"], q_t)

            summary = {
                "features_file": rec["path"],
                "checkpoint": checkpoint,
                "sequence": sequence,
                "stage": stage,
                "trained_language": trained_language,
                "layer": args.layer,
                "reference_lang_rank": int(q_ref.shape[1]),
                "previous_stage": int(prev_stage),
                **prefix_dict(base_drift, "base_"),
                **prefix_dict(step_drift, "step_"),
                **global_energy,
                **prefix_dict(base_subspace, "base_"),
                **prefix_dict(step_subspace, "step_"),
            }
            summary_rows.append(summary)

            for lang in unique_langs:
                idx = torch.tensor(
                    [
                        i
                        for i, (s, l) in enumerate(zip(splits, languages))
                        if s == "probe_test" and l == lang
                    ],
                    dtype=torch.long,
                    device=device,
                )

                anchor_stage = find_anchor_stage(seq_records, stage, lang)
                anchor_rec = seq_records[anchor_stage]
                anchor_delta = xt - anchor_rec["x"]
                anchor_drift = drift_metrics(anchor_delta[idx], q_ref)
                anchor_subspace = subspace_similarity(anchor_rec["q"], q_t)

                base_lang_drift = drift_metrics(base_delta[idx], q_ref)
                em = representation_energy_metrics(xt[idx], q_ref)

                lang_rows.append({
                    "features_file": rec["path"],
                    "checkpoint": checkpoint,
                    "sequence": sequence,
                    "stage": stage,
                    "trained_language": trained_language,
                    "layer": args.layer,
                    "language": lang,
                    "n": int(idx.numel()),
                    "anchor_stage": int(anchor_stage),
                    **prefix_dict(base_lang_drift, "base_"),
                    **prefix_dict(anchor_drift, "anchor_"),
                    **em,
                    **prefix_dict(base_subspace, "base_"),
                    **prefix_dict(anchor_subspace, "anchor_"),
                })

            print(
                f"{sequence} stage={stage:<2} trained={trained_language:<4} "
                f"base_drift={base_drift['mean_total_drift_l2']:.4f} "
                f"step_drift={step_drift['mean_total_drift_l2']:.4f} "
                f"base_overlap={base_subspace['lang_subspace_overlap']:.4f} "
                f"step_overlap={step_subspace['lang_subspace_overlap']:.4f}"
            )

    summary_fields = sorted({k for r in summary_rows for k in r})
    with (out / "subspace_tracking_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        w.writerows(summary_rows)

    lang_fields = sorted({k for r in lang_rows for k in r})
    with (out / "subspace_tracking_by_language.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.DictWriter(f, fieldnames=lang_fields)
        w.writeheader()
        w.writerows(lang_rows)

    print(f"Saved: {out / 'reference_inlp_language_subspace.pt'}")
    print(f"Saved: {out / 'subspace_tracking_summary.csv'}")
    print(f"Saved: {out / 'subspace_tracking_by_language.csv'}")


if __name__ == "__main__":
    main()
