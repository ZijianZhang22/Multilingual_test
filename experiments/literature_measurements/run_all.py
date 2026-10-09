#!/usr/bin/env python3
"""Resume-safe one-command orchestrator for multilingual literature measurements.

Modes:
  pilot = fit a clean Drift/Transfer/ISR core, aligned semantic patching,
          bidirectional Wiki LM interventions, and unified summary.
  full  = pilot + affine geometry, LayerMoE attention similarity, aligned
          CKA, held-out transfer probes, mean-centered retrieval, whole-layer
          patching and a descriptive cross-metric association table.

No model training or checkpoint modification is performed. Commands are run
sequentially with captured logs and explicit output validation.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable


@dataclass
class Step:
    name: str
    argv: list[str]
    outputs: tuple[Path, ...]


def command(module: str, *args: object) -> list[str]:
    return [PYTHON, "-m", module, *(str(arg) for arg in args)]


def build_plan(a: argparse.Namespace) -> list[Step]:
    """Construct all steps without GPU, filesystem writes or downloads."""
    out = Path(a.out).expanduser().resolve()
    core_dir = out / "core"
    core = core_dir / "core_subspaces.pt"
    probe = core_dir / "data" / "xnli_probe.jsonl"
    aligned_test = out / "aligned_test.jsonl"
    aligned_fit = out / "aligned_fit.jsonl"
    spaces = ["drift", "transfer", "isr_cov", "isr_multiclass"]
    lang = [a.old_language, a.new_language]
    layers = sorted({int(x) for x in a.layers.replace(",", " ").split()})
    if not layers:
        raise ValueError("Provide one or more layers")

    def add(steps, name, argv, *outputs):
        steps.append(Step(name, argv, tuple(Path(p) for p in outputs)))

    steps = []
    add(steps, "01_core",
        command("experiments.retention_subspace_replication.build_core_subspaces",
            "--anchor_checkpoint", a.anchor, "--adapted_checkpoint", a.adapted,
            "--languages", *a.fit_languages, "--layer", a.core_layer,
            "--rank", a.rank, "--pool", "last",
            "--probe_train_per_lang", a.train_per_lang,
            "--probe_test_per_lang", a.test_per_lang,
            "--aligned_examples", a.aligned_fit_examples,
            "--extract_batch", a.batch_size,
            "--out_dir", core_dir),
        core, probe)
    add(steps, "02_aligned_test",
        [PYTHON, "invariance/prepare_aligned_xnli.py",
         "--languages", *lang, "--split", "test",
         "--n_examples", str(a.aligned_test_examples),
         "--out_file", str(aligned_test)],
        aligned_test)
    add(steps, "03_semantic",
        command("experiments.literature_measurements.semantic_subspace_patch",
            "--checkpoint", a.adapted, "--core_file", core,
            "--eval_jsonl", aligned_test, "--languages", *lang,
            "--spaces", *spaces, "--n_targets_per_language", a.targets_per_lang,
            "--alpha", a.alpha, "--out_dir", out / "semantic"),
        out / "semantic" / "semantic_patching_summary.csv")
    add(steps, "04_bidirectional",
        command("experiments.literature_measurements.bidirectional_subspace",
            "--anchor_checkpoint", a.anchor, "--adapted_checkpoint", a.adapted,
            "--core_file", core, "--data_dir", a.data_dir,
            "--old_language", a.old_language, "--new_language", a.new_language,
            "--spaces", *spaces, "--max_blocks", a.max_blocks,
            "--batch_size", a.batch_size, "--alpha", a.alpha,
            "--out_dir", out / "bidirectional"),
        out / "bidirectional" / "bidirectional_subspace_summary.csv")
    probe_summary = []
    if a.mode == "full":
        fit_probe = out / "xnli_fit.jsonl"
        eval_probe = out / "xnli_eval.jsonl"
        add(steps, "05_split_probe",
            command("experiments.literature_measurements.split_probe",
                "--input_jsonl", probe,
                "--fit_jsonl", fit_probe, "--eval_jsonl", eval_probe),
            fit_probe, eval_probe)
        add(steps, "06_affine",
            command("experiments.literature_measurements.affine_projection",
                "--anchor_checkpoint", a.anchor, "--adapted_checkpoint", a.adapted,
                "--fit_jsonl", fit_probe, "--eval_jsonl", eval_probe,
                "--languages", *lang, "--layers", ",".join(map(str, layers)),
                "--batch_size", a.batch_size,
                "--fit_examples_per_lang", a.affine_fit_examples,
                "--eval_examples_per_lang", a.affine_eval_examples,
                "--max_fit_tokens_per_lang", a.affine_tokens,
                "--max_rank", a.affine_rank,
                "--out_dir", out / "affine"),
            out / "affine" / "affine_projection_results.csv",
            out / "affine" / "affine_basis_metadata.csv")

        for ckpt_name, checkpoint in (("anchor", a.anchor), ("adapted", a.adapted)):
            path = out / f"{ckpt_name}_attention.pt"
            add(steps, f"07_attention_{ckpt_name}",
                command("experiments.literature_measurements.extract_attention",
                    "--checkpoint", checkpoint, "--data_file", probe,
                    "--out_file", path, "--layers", ",".join(map(str, layers)),
                    "--source", "attention", "--pool", "tokens",
                    "--split", "probe_test", "--languages", *lang,
                    "--max_rows_per_language", a.similarity_examples,
                    "--max_tokens_per_language", a.similarity_tokens,
                    "--batch_size", a.batch_size),
                path)
        add(steps, "08_similarity",
            command("experiments.literature_measurements.layer_similarity",
                "--anchor_features", out / "anchor_attention.pt",
                "--adapted_features", out / "adapted_attention.pt",
                "--out_csv", out / "layer_similarity.csv",
                "--layers", ",".join(map(str, layers)),
                "--split", "probe_test", "--languages", *lang),
            out / "layer_similarity.csv")

        for ckpt_name, checkpoint in (("anchor", a.anchor), ("adapted", a.adapted)):
            path = out / f"{ckpt_name}_aligned_attention.pt"
            add(steps, f"09_aligned_attention_{ckpt_name}",
                command("experiments.literature_measurements.extract_attention",
                    "--checkpoint", checkpoint, "--data_file", aligned_test,
                    "--out_file", path, "--layers", ",".join(map(str, layers)),
                    "--source", "attention", "--pool", "mean", "--split", "aligned",
                    "--languages", *lang, "--max_rows_per_language",
                    a.aligned_test_examples, "--batch_size", a.batch_size),
                path)
        add(steps, "10_aligned_cka",
            command("experiments.literature_measurements.layer_similarity",
                "--anchor_features", out / "anchor_aligned_attention.pt",
                "--adapted_features", out / "adapted_aligned_attention.pt",
                "--out_csv", out / "aligned_cka.csv", "--split", "aligned",
                "--layers", ",".join(map(str, layers)), "--languages", *lang),
            out / "aligned_cka.csv")

        for ckpt_name, checkpoint in (("anchor", a.anchor), ("adapted", a.adapted)):
            path = out / f"{ckpt_name}_residual_last.pt"
            add(steps, f"11_residual_{ckpt_name}",
                [PYTHON, "invariance/extract_hidden.py", "--checkpoint", checkpoint,
                 "--data_file", str(probe), "--out_file", str(path),
                 "--pool", "last", "--layers", *[str(l) for l in layers],
                 "--batch_size", str(a.batch_size)],
                path)
        add(steps, "12_transfer_probe",
            command("experiments.literature_measurements.transfer_probe",
                "--anchor_features", out / "anchor_residual_last.pt",
                "--adapted_features", out / "adapted_residual_last.pt",
                "--basis_file", core, "--layers", str(a.core_layer),
                "--languages", *a.fit_languages, "--source_language",
                a.old_language, "--rank", a.rank, "--out_csv", out / "transfer_probe.csv"),
            out / "transfer_probe.csv")
        probe_summary = ["--probe_csv", str(out / "transfer_probe.csv")]

        add(steps, "13_layer_patch",
            command("experiments.literature_measurements.layer_patch",
                "--anchor_checkpoint", a.anchor, "--adapted_checkpoint", a.adapted,
                "--data_dir", a.data_dir, "--languages", *lang,
                "--old_language", a.old_language,
                "--layers", ",".join(map(str, layers)),
                "--eval_max_blocks", a.max_blocks, "--batch_size", a.batch_size,
                "--out_csv", out / "layer_patch.csv"),
            out / "layer_patch.csv")
        add(steps, "14_association",
            command("experiments.literature_measurements.association",
                "--similarity_csv", out / "layer_similarity.csv",
                "--patch_csv", out / "layer_patch.csv",
                "--language_a", a.old_language, "--language_b", a.new_language,
                "--old_language", a.old_language, "--out_dir", out / "association"),
            out / "association" / "layer_sharedness_patch_join.csv")

        add(steps, "15_aligned_fit",
            [PYTHON, "invariance/prepare_aligned_xnli.py",
             "--languages", *lang, "--split", "validation",
             "--n_examples", str(a.aligned_fit_examples),
             "--out_file", str(aligned_fit)],
            aligned_fit)
        for ckpt_name, checkpoint in (("anchor", a.anchor), ("adapted", a.adapted)):
            for partition, data_file in (("fit", aligned_fit), ("eval", aligned_test)):
                dest = out / f"{ckpt_name}_aligned_{partition}_residual.pt"
                add(steps, f"16_retrieval_features_{ckpt_name}_{partition}",
                    command("experiments.literature_measurements.extract_attention",
                        "--checkpoint", checkpoint, "--data_file", data_file,
                        "--out_file", dest, "--source", "residual",
                        "--pool", "mean", "--split", "aligned",
                        "--layers", ",".join(map(str, layers)),
                        "--languages", *lang, "--max_rows_per_language",
                        max(a.aligned_fit_examples, a.aligned_test_examples),
                        "--max_tokens_per_language",
                        max(a.aligned_fit_examples, a.aligned_test_examples),
                        "--batch_size", a.batch_size),
                    dest)
        add(steps, "17_retrieval",
            command("experiments.literature_measurements.retrieval",
                "--anchor_fit", out / "anchor_aligned_fit_residual.pt",
                "--anchor_eval", out / "anchor_aligned_eval_residual.pt",
                "--adapted_fit", out / "adapted_aligned_fit_residual.pt",
                "--adapted_eval", out / "adapted_aligned_eval_residual.pt",
                "--languages", *lang, "--layers", ",".join(map(str, layers)),
                "--out_csv", out / "mean_centered_retrieval.csv"),
            out / "mean_centered_retrieval.csv")

    add(steps, "18_unified_summary",
        command("experiments.literature_measurements.subspace_functional_summary",
            "--semantic_csv", out / "semantic" / "semantic_patching_summary.csv",
            "--bidirectional_csv", out / "bidirectional" / "bidirectional_subspace_summary.csv",
            *probe_summary, "--old_language", a.old_language,
            "--new_language", a.new_language,
            "--out_dir", out / "summary"),
        out / "summary" / "subspace_functional_summary.csv")
    return steps


def safe_filename(name: str) -> str:
    return "".join(ch for ch in name if ch.isascii() and (ch.isalnum() or ch in "-_"))


def make_archive(out: Path, include_pt=False) -> Path:
    if not out.is_dir():
        raise FileNotFoundError(out)
    archive = out.parent / (out.name + "_results.tar.gz")
    allowed = {".csv", ".json", ".log", ".md", ".txt", ".yaml", ".yml"}
    if include_pt:
        allowed.add(".pt")
    with tarfile.open(archive, "w:gz") as tar:
        for f in sorted(out.rglob("*")):
            if f.is_file() and f.suffix.lower() in allowed:
                tar.add(f, arcname=str(Path(out.name) / f.relative_to(out)),
                        recursive=False)
    return archive


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--anchor", help="Local Stage-1 Anchor checkpoint directory")
    p.add_argument("--adapted", help="Local Stage-2 Adapted checkpoint directory")
    p.add_argument("--out", default="mechanism_runs/literature_measurements_3b")
    p.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    p.add_argument("--core-layer", type=int, default=20)
    p.add_argument("--layers", default="6,12,20,24",
                   help="Additional layers for full mode; MUST include core-layer")
    p.add_argument("--data-dir", default="invariance_data/wiki")
    p.add_argument("--old-language", default="en")
    p.add_argument("--new-language", default="zh")
    p.add_argument("--fit-languages", nargs="+", default=["en", "zh", "fr", "de", "es"])
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--train-per-lang", type=int, default=350)
    p.add_argument("--test-per-lang", type=int, default=150)
    p.add_argument("--aligned-fit-examples", type=int, default=200)
    p.add_argument("--aligned-test-examples", type=int, default=120)
    p.add_argument("--targets-per-lang", type=int, default=6)
    p.add_argument("--max-blocks", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--affine-fit-examples", type=int, default=120)
    p.add_argument("--affine-eval-examples", type=int, default=60)
    p.add_argument("--affine-tokens", type=int, default=4096)
    p.add_argument("--affine-rank", type=int, default=128)
    p.add_argument("--similarity-examples", type=int, default=120)
    p.add_argument("--similarity-tokens", type=int, default=1200)
    p.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    p.add_argument("--no-resume", action="store_true",
                   help="Rerun completed steps; do not reuse previous stage markers")
    p.add_argument("--archive-only", action="store_true",
                   help="Package completed outputs, do not execute experiments")
    p.add_argument("--include-pt", action="store_true",
                   help="Also include potentially large PyTorch feature files in archive")
    a = p.parse_args(argv)
    if a.archive_only:
        return a
    if not a.anchor or not a.adapted:
        p.error("--anchor and --adapted are required (except --archive-only)")
    if a.old_language == a.new_language:
        p.error("old and new language must differ")
    if not {a.old_language, a.new_language}.issubset(set(a.fit_languages)):
        p.error("both evaluated languages must be included in --fit-languages")
    if not 0 < a.alpha <= 1 or a.rank <= 0 or a.batch_size <= 0:
        p.error("alpha must be in (0,1], rank and batch-size must be positive")
    if any(x < 3 for x in (a.train_per_lang, a.test_per_lang,
                           a.aligned_fit_examples, a.aligned_test_examples,
                           a.targets_per_lang)):
        p.error("sample counts must be >=3")
    if a.max_blocks < 1 or a.core_layer < 1:
        p.error("max-blocks/core-layer must be positive")
    if a.mode == "full":
        layers = {int(s) for s in a.layers.replace(",", " ").split()}
        if a.core_layer not in layers:
            p.error("--layers must contain --core-layer when mode=full")
    return a


def run_steps(a, steps):
    out = Path(a.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    conf_path = out / "run_manifest.json"
    excluded = {"dry_run", "no_resume", "archive_only", "include_pt"}
    config = {k: v for k, v in vars(a).items() if k not in excluded}
    if conf_path.exists():
        before = json.loads(conf_path.read_text(encoding="utf-8"))
        if before.get("config") != config:
            raise RuntimeError(
                f"Output directory {out} contains a different configuration. "
                "Choose a fresh --out directory to prevent silent result mixing."
            )
    else:
        conf_path.write_text(json.dumps({"config": config, "stage_names": [s.name for s in steps]},
                                         indent=2, ensure_ascii=False) + "\n")
    stage_dir, log_dir = out / ".completed", out / "logs"
    stage_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    for index, step in enumerate(steps, 1):
        marker = stage_dir / (safe_filename(step.name) + ".done")
        if not a.no_resume and marker.exists() and all(p.is_file() and p.stat().st_size > 0
                                                       for p in step.outputs):
            print(f"[{index}/{len(steps)}] SKIP {step.name} (completed)", flush=True)
            continue
        marker.unlink(missing_ok=True)
        log_file = log_dir / (safe_filename(step.name) + ".log")
        print(f"[{index}/{len(steps)}] RUN {step.name}", flush=True)
        print("  " + shlex.join(step.argv), flush=True)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"]
                                          if env.get("PYTHONPATH") else "")
        with log_file.open("a", encoding="utf-8") as log:
            log.write(f"\n=== {time.strftime('%Y-%m-%dT%H:%M:%S')} ===\n")
            log.write(shlex.join(step.argv) + "\n")
            log.flush()
            proc = subprocess.Popen(step.argv, cwd=ROOT, env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding="utf-8", errors="replace", bufsize=1)
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
            code = proc.wait()
        if code:
            raise RuntimeError(
                f"FAILED {step.name} with exit {code}. See log: {log_file}"
            )
        missing = [str(path) for path in step.outputs
                   if not path.is_file() or path.stat().st_size == 0]
        if missing:
            raise RuntimeError(f"{step.name} exited 0 but outputs missing: {missing}")
        marker.write_text("success\n", encoding="utf-8")
    archive = make_archive(out, a.include_pt)
    print(f"\nAll {len(steps)} stages complete. Results archive: {archive}", flush=True)


def main(argv=None):
    a = parse_args(argv)
    if a.archive_only:
        archive = make_archive(Path(a.out).expanduser().resolve(), a.include_pt)
        print(f"Results archive: {archive}")
        return
    steps = build_plan(a)
    if a.dry_run:
        print(f"MODE={a.mode}, stages={len(steps)}")
        for step in steps:
            print(f"{step.name}\n    {shlex.join(step.argv)}")
        return
    for ckpt in (a.anchor, a.adapted):
        if not Path(ckpt).is_dir() or not (Path(ckpt)/"config.json").is_file():
            raise FileNotFoundError(f"Expected local checkpoint directory with config.json: {ckpt}")
    for lang in (a.old_language, a.new_language):
        pt = Path(a.data_dir) / (lang + "_val.pt")
        if not pt.is_file():
            raise FileNotFoundError(f"Missing pre-tokenized Wiki validation blocks: {pt}")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required. Use --dry-run to preview the plan.")
    run_steps(a, steps)


if __name__ == "__main__":
    main()
