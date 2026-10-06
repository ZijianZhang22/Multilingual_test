import argparse
from pathlib import Path
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", required=True)
    ap.add_argument("--data_file", default="invariance_data/xnli_probe.jsonl")
    ap.add_argument("--out_dir", default="invariance_features/sequence_checkpoints")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--pool", choices=["mean", "last"], default="mean")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = sorted(
        p for p in runs_dir.glob("*/*")
        if p.is_dir() and p.name.startswith("stage")
    )
    if not checkpoints:
        raise FileNotFoundError(
            f"No stage checkpoints found under {runs_dir}. "
            "Expected paths such as en__zh/stage0_base."
        )

    for ckpt in checkpoints:
        sequence = ckpt.parent.name
        out_file = out_dir / sequence / f"{ckpt.name}.pt"
        out_file.parent.mkdir(parents=True, exist_ok=True)

        if out_file.exists():
            print(f"SKIP existing: {out_file}")
            continue

        cmd = [
            args.python,
            str(Path(__file__).with_name("extract_hidden.py")),
            "--checkpoint", str(ckpt),
            "--data_file", args.data_file,
            "--out_file", str(out_file),
            "--layers", str(args.layer),
            "--batch_size", str(args.batch_size),
            "--max_length", str(args.max_length),
            "--pool", args.pool,
        ]
        print("RUN:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    print(f"Done. Features saved under {out_dir}")


if __name__ == "__main__":
    main()
