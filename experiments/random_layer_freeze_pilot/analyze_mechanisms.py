#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


def run(cmd):
    print("\n>>>", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(x) for x in cmd], check=True)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError("Cannot find transformer blocks")


def snapshot_anchor_layers(checkpoint, layers):
    print(f"Loading anchor on CPU for parameter-drift reference: {checkpoint}")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    blocks = get_layers(model)
    out = {}
    for layer in layers:
        block = blocks[layer - 1]
        out[layer] = {k: v.detach().cpu().float().clone() for k, v in block.state_dict().items()}
    del model
    return out


def parameter_drift(checkpoint, anchor_states, layers):
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    blocks = get_layers(model)
    rows = []
    for layer in layers:
        post = blocks[layer - 1].state_dict()
        sq_delta = 0.0
        sq_anchor = 0.0
        abs_delta = 0.0
        count = 0
        changed = 0
        dot = 0.0
        sq_post = 0.0
        for k, a in anchor_states[layer].items():
            p = post[k].detach().cpu().float()
            d = p - a
            sq_delta += float((d * d).sum())
            sq_anchor += float((a * a).sum())
            sq_post += float((p * p).sum())
            dot += float((a * p).sum())
            abs_delta += float(d.abs().sum())
            count += d.numel()
            changed += int((d.abs() > 1e-8).sum())
        rows.append({
            "layer": layer,
            "relative_parameter_drift": math.sqrt(sq_delta) / max(math.sqrt(sq_anchor), 1e-12),
            "mean_abs_parameter_delta": abs_delta / max(count, 1),
            "changed_parameter_fraction": changed / max(count, 1),
            "anchor_post_parameter_cosine": dot / max(math.sqrt(sq_anchor * sq_post), 1e-12),
        })
    del model
    return rows


def parse_condition_layer(name):
    m = re.match(r"layer(\d+)_freeze(\d+)$", name)
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot_dir", default="random_layer_freeze_runs/en_to_zh_seed0")
    ap.add_argument("--anchor_checkpoint", default="invariance_runs/sequence_seed0/en__zh/stage1_en")
    ap.add_argument("--data_dir", default="invariance_data/wiki")
    ap.add_argument("--old_language", default="en")
    ap.add_argument("--new_language", default="zh")
    ap.add_argument("--layers", nargs="+", type=int, default=[12, 20, 24])
    ap.add_argument("--probe_file", default="invariance_data/xnli_freeze_pilot.jsonl")
    ap.add_argument("--probe_per_lang", type=int, default=100)
    ap.add_argument("--gradient_max_batches", type=int, default=4)
    ap.add_argument("--gradient_max_blocks", type=int, default=32)
    ap.add_argument("--skip_parameter_drift", action="store_true")
    ap.add_argument("--skip_cka", action="store_true")
    ap.add_argument("--skip_gradient", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    pilot = Path(args.pilot_dir)
    ckpt_root = pilot / "checkpoints"
    mech = pilot / "mechanism"
    mech.mkdir(parents=True, exist_ok=True)

    result_file = pilot / "results.csv"
    if not result_file.exists():
        raise FileNotFoundError(f"Missing {result_file}; run the pilot first.")
    behavior = {r["condition"]: r for r in read_csv(result_file)}

    if not ckpt_root.exists():
        raise FileNotFoundError(
            f"Missing {ckpt_root}. Re-run pilot with --save_checkpoints "
            "(the provided run_small_pilot.sh now enables this by default)."
        )
    conditions = sorted([p.name for p in ckpt_root.iterdir() if p.is_dir()])
    if not conditions:
        raise RuntimeError(f"No checkpoints found under {ckpt_root}")
    print("Conditions:", conditions)

    param_rows = []
    if not args.skip_parameter_drift:
        anchor_states = snapshot_anchor_layers(args.anchor_checkpoint, args.layers)
        for cond in conditions:
            print(f"[parameter drift] {cond}")
            rows = parameter_drift(ckpt_root / cond, anchor_states, args.layers)
            for r in rows:
                r["condition"] = cond
                param_rows.append(r)
        write_csv(mech / "parameter_drift.csv", param_rows)

    cka_rows = []
    if not args.skip_cka:
        probe = Path(args.probe_file)
        if not probe.exists():
            run([
                sys.executable, root / "invariance/prepare_xnli.py",
                "--languages", args.old_language, args.new_language,
                "--train_per_lang", str(args.probe_per_lang),
                "--test_per_lang", str(args.probe_per_lang),
                "--seed", "2026",
                "--out_file", probe,
            ])
        feat_dir = mech / "features"
        feat_dir.mkdir(exist_ok=True)
        anchor_feat = feat_dir / "anchor.pt"
        if not anchor_feat.exists():
            run([
                sys.executable, root / "invariance/extract_hidden.py",
                "--checkpoint", args.anchor_checkpoint,
                "--data_file", probe,
                "--out_file", anchor_feat,
                "--layers", *map(str, args.layers),
                "--batch_size", "16",
            ])
        for cond in conditions:
            post_feat = feat_dir / f"{cond}.pt"
            if not post_feat.exists():
                run([
                    sys.executable, root / "invariance/extract_hidden.py",
                    "--checkpoint", ckpt_root / cond,
                    "--data_file", probe,
                    "--out_file", post_feat,
                    "--layers", *map(str, args.layers),
                    "--batch_size", "16",
                ])
            out_csv = mech / "cka" / f"{cond}.csv"
            run([
                sys.executable, root / "invariance/analyze_layerwise_cka.py",
                "--anchor_features", anchor_feat,
                "--post_features", post_feat,
                "--layers", *map(str, args.layers),
                "--out_file", out_csv,
            ])
            for r in read_csv(out_csv):
                r["condition"] = cond
                cka_rows.append(r)
        write_csv(mech / "cka_all.csv", cka_rows)

    grad_rows = []
    if not args.skip_gradient:
        for cond in conditions:
            out_csv = mech / "gradient" / f"{cond}.csv"
            run([
                sys.executable, root / "invariance/analyze_gradient_interference.py",
                "--checkpoint", ckpt_root / cond,
                "--data_dir", args.data_dir,
                "--old_language", args.old_language,
                "--new_language", args.new_language,
                "--layers", *map(str, args.layers),
                "--batch_size", "2",
                "--max_batches", str(args.gradient_max_batches),
                "--max_blocks", str(args.gradient_max_blocks),
                "--seed", "0",
                "--out_file", out_csv,
            ])
            for r in read_csv(out_csv):
                r["condition"] = cond
                grad_rows.append(r)
        write_csv(mech / "gradient_all.csv", grad_rows)

    # Merge into one long-form table: one row per condition x layer.
    P = {(r["condition"], int(r["layer"])): r for r in param_rows}
    C = {(r["condition"], int(r["layer"]), r["language"]): r for r in cka_rows}
    G = {(r["condition"], int(r["layer"])): r for r in grad_rows}
    summary = []
    for cond in conditions:
        b = behavior.get(cond, {})
        target = parse_condition_layer(cond)
        for layer in args.layers:
            p = P.get((cond, layer), {})
            ca = C.get((cond, layer, "__all__"), {})
            co = C.get((cond, layer, args.old_language), {})
            cn = C.get((cond, layer, args.new_language), {})
            g = G.get((cond, layer), {})
            summary.append({
                "condition": cond,
                "target_layer": "" if target is None else target,
                "analysis_layer": layer,
                "is_target_layer": int(target == layer) if target is not None else 0,
                "forgetting": b.get("forgetting", ""),
                "new_language_gain": b.get("new_language_gain", ""),
                "relative_parameter_drift": p.get("relative_parameter_drift", ""),
                "changed_parameter_fraction": p.get("changed_parameter_fraction", ""),
                "cka_all": ca.get("linear_cka", ""),
                f"cka_{args.old_language}": co.get("linear_cka", ""),
                f"cka_{args.new_language}": cn.get("linear_cka", ""),
                "relative_hidden_l2_all": ca.get("relative_l2_drift", ""),
                "old_new_gradient_cosine": g.get("mean_gradient_cosine", ""),
                "negative_gradient_batch_fraction": g.get("negative_batch_fraction", ""),
                "old_gradient_norm": g.get("old_gradient_norm", ""),
                "new_gradient_norm": g.get("new_gradient_norm", ""),
            })
    write_csv(mech / "mechanism_summary.csv", summary)

    print("\n=== Mechanism outputs ===")
    for x in [
        mech / "parameter_drift.csv",
        mech / "cka_all.csv",
        mech / "gradient_all.csv",
        mech / "mechanism_summary.csv",
    ]:
        if x.exists():
            print(x)
    print("\nInterpretation:")
    print("1) Parameter drift: did freezing actually reduce movement in the targeted layer?")
    print("2) CKA/RMS: did the targeted layer preserve old representations more than other layers?")
    print("3) Gradient cosine: did adaptation change old-vs-new gradient conflict/alignment?")


if __name__ == "__main__":
    main()
