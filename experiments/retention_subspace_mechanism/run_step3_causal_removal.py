#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.random_layer_freeze_pilot.run_pilot import get_layers, load_model  # noqa: E402
from invariance.train_sequence import evaluate, load_blocks, make_loader  # noqa: E402


def replace_hidden(output, new_hidden):
    if isinstance(output, tuple):
        return (new_hidden, *output[1:])
    return new_hidden


@torch.no_grad()
def evaluate_removed(
    model,
    blocks,
    batch_size,
    device,
    use_bf16,
    *,
    layer_no,
    basis,
    center,
    strength,
    scale=1.0,
):
    layer = get_layers(model)[layer_no - 1]
    q = basis.to(device=device, dtype=torch.float32)
    c = center.to(device=device, dtype=torch.float32)

    stats = {"removed_sq": 0.0, "centered_sq": 0.0, "n": 0}

    def hook(_module, _inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        hf = h.float()
        centered = hf - c.view(1, 1, -1)
        projected = (centered @ q) @ q.T
        perturb = strength * scale * projected
        new_h = hf - perturb

        stats["removed_sq"] += float(perturb.pow(2).sum())
        stats["centered_sq"] += float(centered.pow(2).sum())
        stats["n"] += centered.numel()

        return replace_hidden(output, new_h.to(dtype=h.dtype))

    handle = layer.register_forward_hook(hook)
    try:
        loss = evaluate(model, blocks, batch_size, device, use_bf16)
    finally:
        handle.remove()

    fraction = stats["removed_sq"] / max(stats["centered_sq"], 1e-30)
    return loss, fraction


def main():
    ap = argparse.ArgumentParser(
        description="Step 3: matched-rank causal removal of Layer-20 subspaces."
    )
    ap.add_argument(
        "--anchor_checkpoint",
        default="invariance_runs/sequence_seed0/en__zh/stage1_en",
    )
    ap.add_argument(
        "--adapted_checkpoint",
        default="mechanism_runs/step1_layer_lambda_sweep/checkpoints/full_ft",
    )
    ap.add_argument(
        "--subspace_file",
        default="mechanism_runs/step2_layer20_subspaces/layer20_subspaces.pt",
    )
    ap.add_argument("--data_dir", default="invariance_data/wiki")
    ap.add_argument("--languages", nargs="+", default=["en", "zh"])
    ap.add_argument(
        "--strengths",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 1.0],
        help="Fraction of the selected projected component removed.",
    )
    ap.add_argument("--eval_max_blocks", type=int, default=128)
    ap.add_argument("--eval_batch", type=int, default=8)
    ap.add_argument(
        "--states",
        nargs="+",
        default=["anchor", "adapted"],
        choices=["anchor", "adapted"],
        help="Evaluate causal removal on the anchor, adapted model, or both.",
    )
    ap.add_argument(
        "--out_dir",
        default="mechanism_runs/step3_causal_removal",
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    if any(s <= 0 or s > 1 for s in args.strengths):
        raise ValueError("--strengths must be in (0,1].")

    device = torch.device("cuda")
    use_bf16 = torch.cuda.is_bf16_supported()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    payload = torch.load(args.subspace_file, map_location="cpu")
    layer_no = int(payload["layer"])
    center = payload["center"].float()
    subspaces = {k: v.float() for k, v in payload["subspaces"].items()}
    real_subspaces = payload.get(
        "real_subspaces",
        [k for k in subspaces if not k.startswith("random_")],
    )
    matched_controls = payload.get(
        "matched_random_controls",
        {k: f"random_{k}" for k in real_subspaces},
    )

    val = {}
    for lang in args.languages:
        blocks = load_blocks(Path(args.data_dir) / f"{lang}_val.pt")
        if args.eval_max_blocks > 0:
            blocks = blocks[: args.eval_max_blocks]
        val[lang] = blocks

    ckpts = {
        "anchor": args.anchor_checkpoint,
        "adapted": args.adapted_checkpoint,
    }

    rows = []
    for state in args.states:
        print(f"\n===== model_state={state} =====")
        model = load_model(ckpts[state], device, use_bf16)
        model.eval()

        baseline = {}
        for lang in args.languages:
            baseline[lang] = evaluate(
                model, val[lang], args.eval_batch, device, use_bf16
            )
            print(f"[baseline] {state} {lang} loss={baseline[lang]:.6f}")
            rows.append(
                {
                    "model_state": state,
                    "subspace": "none",
                    "matched_control": "",
                    "rank": 0,
                    "strength": 0.0,
                    "language": lang,
                    "loss": baseline[lang],
                    "loss_delta_vs_no_intervention": 0.0,
                    "removed_energy_fraction": 0.0,
                }
            )

        for name, q in subspaces.items():
            match = ""
            if name in matched_controls:
                match = matched_controls[name]
            elif name in matched_controls.values():
                match = next(
                    real for real, ctrl in matched_controls.items()
                    if ctrl == name
                )

            for strength in args.strengths:
                for lang in args.languages:
                    loss, removed_fraction = evaluate_removed(
                        model,
                        val[lang],
                        args.eval_batch,
                        device,
                        use_bf16,
                        layer_no=layer_no,
                        basis=q,
                        center=center,
                        strength=strength,
                    )
                    delta = loss - baseline[lang]
                    rows.append(
                        {
                            "model_state": state,
                            "subspace": name,
                            "matched_control": match,
                            "rank": int(q.shape[1]),
                            "strength": strength,
                            "language": lang,
                            "loss": loss,
                            "loss_delta_vs_no_intervention": delta,
                            "removed_energy_fraction": removed_fraction,
                        }
                    )
                    print(
                        f"[remove] {state:7s} {name:18s} "
                        f"rank={q.shape[1]:>3d} beta={strength:.2f} "
                        f"lang={lang} delta={delta:+.6f} "
                        f"energy={removed_fraction:.4f}"
                    )

        del model
        torch.cuda.empty_cache()

    results_file = out / "removal_results.csv"
    with results_file.open("w", newline="") as f:
        fields = list(rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # Pair every real subspace against its matched-rank random control.
    paired = []
    for state in args.states:
        for real in real_subspaces:
            random_name = matched_controls[real]
            for strength in args.strengths:
                for lang in args.languages:
                    rr = next(
                        r for r in rows
                        if r["model_state"] == state
                        and r["subspace"] == real
                        and r["strength"] == strength
                        and r["language"] == lang
                    )
                    rc = next(
                        r for r in rows
                        if r["model_state"] == state
                        and r["subspace"] == random_name
                        and r["strength"] == strength
                        and r["language"] == lang
                    )
                    paired.append(
                        {
                            "model_state": state,
                            "real_subspace": real,
                            "random_control": random_name,
                            "rank": rr["rank"],
                            "strength": strength,
                            "language": lang,
                            "real_loss_delta": rr["loss_delta_vs_no_intervention"],
                            "random_loss_delta": rc["loss_delta_vs_no_intervention"],
                            "causal_excess_loss_vs_random": (
                                rr["loss_delta_vs_no_intervention"]
                                - rc["loss_delta_vs_no_intervention"]
                            ),
                            "real_removed_energy_fraction": rr["removed_energy_fraction"],
                            "random_removed_energy_fraction": rc["removed_energy_fraction"],
                        }
                    )

    paired_file = out / "matched_random_comparison.csv"
    with paired_file.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(paired[0].keys()))
        w.writeheader()
        w.writerows(paired)

    meta = {
        "layer": layer_no,
        "anchor_checkpoint": args.anchor_checkpoint,
        "adapted_checkpoint": args.adapted_checkpoint,
        "subspace_file": args.subspace_file,
        "languages": args.languages,
        "strengths": args.strengths,
        "real_subspaces": real_subspaces,
        "matched_random_controls": matched_controls,
        "interpretation": (
            "Positive causal_excess_loss_vs_random means removing the real "
            "subspace hurts more than removing an equally ranked random subspace."
        ),
    }
    (out / "manifest.json").write_text(json.dumps(meta, indent=2))

    print("\n=== Matched-random causal excess at beta=1.0 ===")
    for r in paired:
        if abs(float(r["strength"]) - 1.0) < 1e-12:
            print(
                f"{r['model_state']:7s} {r['real_subspace']:9s} "
                f"{r['language']} excess={r['causal_excess_loss_vs_random']:+.6f}"
            )
    print(f"\nSaved: {results_file}")
    print(f"Saved: {paired_file}")


if __name__ == "__main__":
    main()
