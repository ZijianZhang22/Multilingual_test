#!/usr/bin/env python3
"""Split prepared XNLI probe rows into strictly separated fitting/test JSONLs."""
import argparse
import json
from pathlib import Path

from experiments.literature_measurements.core import rows_from_jsonl, ensure_disjoint


def split_rows(rows):
    fit = [row for row in rows if row.get("split") == "probe_train"]
    eval_rows = [row for row in rows if row.get("split") == "probe_test"]
    if not fit or not eval_rows:
        raise ValueError("Need both probe_train and probe_test")
    ensure_disjoint(fit, eval_rows)
    return fit, eval_rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_jsonl", required=True)
    p.add_argument("--fit_jsonl", required=True)
    p.add_argument("--eval_jsonl", required=True)
    a = p.parse_args()
    rows = rows_from_jsonl(a.input_jsonl)
    fit, eva = split_rows(rows)
    for dest, subset in ((a.fit_jsonl, fit), (a.eval_jsonl, eva)):
        path = Path(dest)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                for row in subset), encoding="utf-8")
        print(f"Saved {len(subset)} examples: {dest}")


if __name__ == "__main__":
    main()
