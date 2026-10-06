import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


LABELS = {0: "entailment", 1: "neutral", 2: "contradiction"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--languages", nargs="+", default=["en", "zh", "fr"])
    ap.add_argument("--split", default="validation", choices=["validation", "test"])
    ap.add_argument("--n_examples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out_file", default="invariance_data/xnli_aligned.jsonl")
    args = ap.parse_args()

    datasets = {
        lang: load_dataset("facebook/xnli", lang, split=args.split)
        for lang in args.languages
    }
    lengths = {lang: len(ds) for lang, ds in datasets.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"XNLI language splits are not aligned in length: {lengths}")

    indices = list(range(next(iter(lengths.values()))))
    random.Random(args.seed).shuffle(indices)
    indices = indices[: min(args.n_examples, len(indices))]

    rows = []
    for pair_id, idx in enumerate(indices):
        labels = [int(datasets[lang][idx]["label"]) for lang in args.languages]
        if len(set(labels)) != 1 or labels[0] not in LABELS:
            continue
        for lang in args.languages:
            ex = datasets[lang][idx]
            rows.append({
                "example_id": f"aligned:{args.split}:{idx}:{lang}",
                "pair_id": f"{args.split}:{idx}",
                "language": lang,
                "split": "aligned",
                "label": labels[0],
                "label_name": LABELS[labels[0]],
                "premise": ex["premise"],
                "hypothesis": ex["hypothesis"],
            })

    out = Path(args.out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        f"Saved {len(rows)} rows = {len(rows)//len(args.languages)} aligned semantic items "
        f"for {args.languages} to {out}"
    )


if __name__ == "__main__":
    main()
