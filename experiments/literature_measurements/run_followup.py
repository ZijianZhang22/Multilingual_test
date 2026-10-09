#!/usr/bin/env python3
"""Second-round causal pilot on EXISTING 3B checkpoints; never retrain or overwrite Full.

Dedicated followup folder, resumable subprocesses, independent α/energy
intervention comparisons and per-batch paired loss output.
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from experiments.literature_measurements.run_all import ROOT, make_archive


def check_paths(root):
    paths = {
        "anchor":root/"training"/"anchor",
        "adapted":root/"training"/"adapted",
        "core":root/"analysis"/"core"/"core_subspaces.pt",
        "test":root/"analysis"/"aligned_test.jsonl",
        "wiki":root/"wiki",
    }
    for k, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {k}: {path}")
    for k in ("anchor", "adapted"):
        if not (paths[k]/"config.json").exists():
            raise FileNotFoundError(f"Checkpoint not complete: {paths[k]}")
    for lang in ("en", "zh"):
        if not (paths["wiki"]/f"{lang}_val.pt").exists():
            raise FileNotFoundError(f"Missing Wiki validation data: {lang}")
    return paths


def plan(a):
    source = Path(a.source).expanduser().resolve()
    paths = check_paths(source)
    out = Path(a.out).expanduser().resolve()
    if out == source or out == source/"analysis":
        raise ValueError("Followup output cannot overwrite the original")
    # NLI interventions are relatively expensive, so run one stronger, powered
    # semantic test on 20-40 targets before sweeping strengths.
    steps = []
    def add(name, subdir, *args):
        target = out/subdir
        steps.append((name, [sys.executable,"-m",
                       "experiments.literature_measurements."+name,
                       *map(str,args), "--out_dir",str(target)],target))
    for energy, alpha in (("matched",1.0),("raw",1.0)):
        add("semantic_subspace_patch",f"semantic_{energy}_a{alpha:g}",
            "--checkpoint",paths["adapted"],"--core_file",paths["core"],
            "--eval_jsonl",paths["test"],"--languages","en","zh",
            "--n_targets_per_language",a.semantic_targets_per_lang,
            "--spaces","drift","transfer","isr_cov","isr_multiclass",
            "--alpha",alpha,"--energy_modes",energy,"--seed",a.seed)
    for energy, alpha in (("matched",0.25),("matched",0.5),("matched",1.0),("raw",1.0)):
        add("bidirectional_subspace",f"bidirectional_{energy}_a{alpha:g}",
            "--anchor_checkpoint",paths["anchor"],
            "--adapted_checkpoint",paths["adapted"],
            "--core_file",paths["core"],"--data_dir",paths["wiki"],
            "--spaces","drift","transfer","isr_cov","isr_multiclass",
            "--old_language","en","--new_language","zh",
            "--max_blocks",a.max_blocks,"--batch_size",a.batch_size,
            "--alpha",alpha,"--energy_mode",energy)
    return out, steps


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",default="/workspace/literature_3b_seed0")
    p.add_argument("--out",default="/workspace/literature_3b_followup")
    p.add_argument("--semantic-targets-per-lang",type=int,default=30)
    p.add_argument("--max-blocks",type=int,default=128)
    p.add_argument("--batch-size",type=int,default=2)
    p.add_argument("--seed",type=int,default=2027,
                   help="Fresh target/donor selection from existing held-out test")
    p.add_argument("--dry-run",action="store_true")
    p.add_argument("--archive-only",action="store_true")
    a=p.parse_args()
    if a.archive_only:
        print(make_archive(Path(a.out).expanduser().resolve()))
        return
    if a.semantic_targets_per_lang<3 or a.max_blocks<1 or a.batch_size<1:
        p.error("Targets >=3 and blocks/batch positive")
    out, steps=plan(a)
    if a.dry_run:
        for i,(_,cmd,_) in enumerate(steps,1):
            print(f"[{i}/{len(steps)}]",shlex.join(cmd))
        return
    out.mkdir(parents=True,exist_ok=True)
    config={"source":str(Path(a.source).resolve()),"targets":a.semantic_targets_per_lang,
            "max_blocks":a.max_blocks,"batch_size":a.batch_size,"seed":a.seed}
    manifest=out/"followup_manifest.json"
    if manifest.exists() and json.loads(manifest.read_text())!=config:
        raise RuntimeError("Followup parameters changed: choose a NEW --out directory")
    manifest.write_text(json.dumps(config,indent=2))
    (out/"logs").mkdir(exist_ok=True)
    for idx,(name,cmd,folder) in enumerate(steps,1):
        is_sem=name=="semantic_subspace_patch"
        expected=folder/("semantic_patching_summary.csv" if is_sem else "bidirectional_subspace_summary.csv")
        detail=folder/("semantic_patching_examples.csv" if is_sem else "bidirectional_batch_level.csv")
        if expected.is_file() and detail.is_file() and expected.stat().st_size and detail.stat().st_size:
            print(f"[{idx}/{len(steps)}] SKIP {folder.name}",flush=True)
            continue
        print(f"[{idx}/{len(steps)}] RUN {folder.name}",flush=True)
        print(shlex.join(cmd),flush=True)
        log=out/"logs"/(folder.name+".log")
        env=os.environ.copy()
        env["PYTHONPATH"]=str(ROOT)+os.pathsep+env.get("PYTHONPATH","")
        env["PYTHONUNBUFFERED"]="1"
        with log.open("a",encoding="utf-8") as f:
            process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in process.stdout:
                print(line,end="",flush=True)
                f.write(line)
                f.flush()
            code=process.wait()
        if code:
            raise RuntimeError(f"Failed {folder.name}: see {log}")
        if not expected.is_file() or not detail.is_file():
            raise RuntimeError(f"Missing outputs from {folder.name}: {expected}, {detail}")
    archive=make_archive(out)
    print(f"\nAll followups complete: {archive}",flush=True)


if __name__=="__main__":
    main()
