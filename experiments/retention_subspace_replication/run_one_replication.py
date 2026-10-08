#!/usr/bin/env python3
"""One complete replication run for one Qwen size and one random seed."""
import argparse, json, subprocess, sys
from pathlib import Path
from transformers import AutoConfig
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent

def run(cmd,log_file):
    cmd=[str(x) for x in cmd]
    print("\n>>>"," ".join(cmd),flush=True)
    log_file.parent.mkdir(parents=True,exist_ok=True)
    with log_file.open("a",encoding="utf-8") as f:
        f.write("\n>>> "+" ".join(cmd)+"\n"); f.flush()
        p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in p.stdout:
            print(line,end="",flush=True); f.write(line); f.flush()
        rc=p.wait()
    if rc!=0: raise subprocess.CalledProcessError(rc,cmd)

def complete_checkpoint(path):
    p=Path(path)
    return (p/"config.json").exists() and any(p.glob("*.safetensors"))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model_name",required=True); ap.add_argument("--model_tag",required=True)
    ap.add_argument("--seed",type=int,required=True)
    ap.add_argument("--data_dir",default="invariance_data/wiki")
    ap.add_argument("--out_root",default="replication_runs/qwen_scale_seed")
    ap.add_argument("--old_language",default="en"); ap.add_argument("--new_language",default="zh")
    ap.add_argument("--relative_layer",type=float,default=20/24)
    ap.add_argument("--rank",type=int,default=64); ap.add_argument("--n_random",type=int,default=8)
    ap.add_argument("--new_train_fraction",type=float,default=0.20)
    ap.add_argument("--lr",type=float,default=2e-5); ap.add_argument("--weight_decay",type=float,default=0.1)
    ap.add_argument("--micro_batch",type=int,default=1); ap.add_argument("--grad_accum",type=int,default=16)
    ap.add_argument("--eval_batch",type=int,default=4); ap.add_argument("--extract_batch",type=int,default=4)
    ap.add_argument("--eval_max_blocks",type=int,default=128); ap.add_argument("--force",action="store_true")
    args=ap.parse_args()
    cfg=AutoConfig.from_pretrained(args.model_name)
    n_layers=int(getattr(cfg,"num_hidden_layers"))
    layer=max(1,min(n_layers,round(n_layers*args.relative_layer)))
    run_dir=Path(args.out_root)/args.model_tag/f"seed{args.seed}"
    train_dir=run_dir/"training"; sub_dir=run_dir/"subspaces"; energy_dir=run_dir/"energy_controls"; log=run_dir/"replication.log"
    run_dir.mkdir(parents=True,exist_ok=True)
    meta={"model_name":args.model_name,"model_tag":args.model_tag,"seed":args.seed,"n_layers":n_layers,
          "relative_layer_target":args.relative_layer,"selected_layer_1based":layer,"selected_relative_depth":layer/n_layers,
          "rank":args.rank,"n_random":args.n_random,"old_language":args.old_language,"new_language":args.new_language}
    (run_dir/"replication_manifest.json").write_text(json.dumps(meta,indent=2))
    print(f"[replication] {args.model_tag} seed={args.seed}: {n_layers} layers -> layer {layer} ({layer/n_layers:.3f})",flush=True)
    anchor=train_dir/"anchor"; adapted=train_dir/"adapted"
    if args.force or not (complete_checkpoint(anchor) and complete_checkpoint(adapted)):
        cmd=[sys.executable,HERE/"train_anchor_adapt.py","--model_name",args.model_name,"--data_dir",args.data_dir,
             "--old_language",args.old_language,"--new_language",args.new_language,"--seed",args.seed,
             "--new_train_fraction",args.new_train_fraction,"--lr",args.lr,"--weight_decay",args.weight_decay,
             "--micro_batch",args.micro_batch,"--grad_accum",args.grad_accum,"--eval_batch",args.eval_batch,
             "--eval_max_blocks",args.eval_max_blocks,"--gradient_checkpointing","--out_dir",train_dir]
        run(cmd,log)
    else: print("[resume] training checkpoints already exist",flush=True)
    subspace_file=sub_dir/"core_subspaces.pt"
    if args.force or not subspace_file.exists():
        cmd=[sys.executable,HERE/"build_core_subspaces.py","--anchor_checkpoint",anchor,"--adapted_checkpoint",adapted,
             "--layer",layer,"--rank",args.rank,"--extract_batch",args.extract_batch,"--out_dir",sub_dir]
        if args.force: cmd.append("--force")
        run(cmd,log)
    else: print("[resume] core subspaces already exist",flush=True)
    summary_file=energy_dir/"energy_matched_summary.csv"
    if args.force or not summary_file.exists():
        cmd=[sys.executable,ROOT/"experiments/retention_subspace_mechanism/run_step6_energy_matched_controls.py",
             "--anchor_checkpoint",anchor,"--adapted_checkpoint",adapted,"--subspace_file",subspace_file,
             "--data_dir",args.data_dir,"--old_language",args.old_language,"--new_language",args.new_language,
             "--subspaces","drift","isr_multiclass","transfer","isr_cov","vicreg","--strengths","0.25","0.5","1.0",
             "--n_random",args.n_random,"--random_seed",7300+100*args.seed,
             "--eval_max_blocks",args.eval_max_blocks,"--eval_batch",args.eval_batch,"--out_dir",energy_dir]
        run(cmd,log)
    else: print("[resume] energy-matched controls already exist",flush=True)
    print(f"\nDONE: {run_dir}",flush=True)
if __name__=="__main__": main()
