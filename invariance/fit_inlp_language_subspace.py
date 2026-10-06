import argparse
from pathlib import Path

import torch

from benchmark_representation_methods import fit_inlp, make_split_indices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_file", required=True)
    ap.add_argument("--out_file", required=True)
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--iters", type=int, default=32)
    ap.add_argument("--classifier_epochs", type=int, default=100)
    ap.add_argument("--classifier_lr", type=float, default=1e-2)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    payload = torch.load(args.features_file, map_location="cpu")
    x = payload["features"][str(args.layer)].float().to(device)
    languages = payload["languages"]
    splits = payload["splits"]

    unique_langs = sorted(set(languages))
    lang_to_id = {lang: i for i, lang in enumerate(unique_langs)}
    language_ids = torch.tensor(
        [lang_to_id[v] for v in languages], dtype=torch.long, device=device
    )
    train_idx = make_split_indices(splits, "probe_train").to(device)

    _, q_lang = fit_inlp(
        x,
        language_ids,
        train_idx,
        iters=args.iters,
        classifier_epochs=args.classifier_epochs,
        classifier_lr=args.classifier_lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )

    out = Path(args.out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "layer": args.layer,
        "inlp_iters": args.iters,
        "input_dim": x.shape[1],
        "language_subspace_basis": q_lang.detach().cpu(),
        "languages": unique_langs,
        "reference_features": args.features_file,
    }, out)
    print(
        f"Saved INLP language subspace rank={q_lang.shape[1]} "
        f"layer={args.layer} to {out}"
    )


if __name__ == "__main__":
    main()
