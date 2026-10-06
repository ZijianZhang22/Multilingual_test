import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


LABELS = {0: "entailment", 1: "neutral", 2: "contradiction"}


def sample_rows(lang, split, n, seed):
    ds = load_dataset("xnli", lang, split=split)
    idx = list(range(len(ds)))
    rng = random.Random(seed)
    rng.shuffle(idx)
    if n is not None:
        idx = idx[: min(n, len(idx))]

    rows = []
    for i in idx:
        ex = ds[i]
        label = int(ex["label"])
        if label not in LABELS:
            continue
        rows.append({
            "example_id": f"{lang}:{split}:{i}",
            "language": lang,
            "split": "probe_train" if split == "validation" else "probe_test",
            "label": label,
            "label_name": LABELS[label],
            "premise": ex["premise"],
            "hypothesis": ex["hypothesis"],
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--languages", nargs="+", default=["en", "zh", "fr"])
    ap.add_argument("--train_per_lang", type=int, default=1200,
                    help="Samples from XNLI validation used to fit probes")
    ap.add_argument("--test_per_lang", type=int, default=1200,
                    help="Samples from XNLI test used only for probe evaluation")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out_file", default="invariance_data/xnli_probe.jsonl")
    args = ap.parse_args()

    rows = []
    for j, lang in enumerate(args.languages):
        rows.extend(sample_rows(lang, "validation", args.train_per_lang, args.seed + 10 * j))
        rows.extend(sample_rows(lang, "test", args.test_per_lang, args.seed + 10 * j + 1))

    out = Path(args.out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = {}
    for row in rows:
        key = (row["language"], row["split"])
        counts[key] = counts.get(key, 0) + 1
    print(f"Saved {len(rows)} examples to {out}")
    for key, value in sorted(counts.items()):
        print(key, value)


if __name__ == "__main__":
    main()
