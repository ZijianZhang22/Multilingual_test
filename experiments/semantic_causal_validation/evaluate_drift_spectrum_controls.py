#!/usr/bin/env python3
"""Run Step-6-compatible 7B rescue across drift spectral bands.

Equal-energy means equal TOTAL intervention squared norm over an evaluation
corpus; no direction is amplified. Target is smallest natural energy per rank.
Run pilot with 16 blocks; then independent confirmatory 128-block evaluation.
"""
import argparse,csv,json,math,sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.random_layer_freeze_pilot.run_pilot import load_model
from experiments.retention_subspace_mechanism.run_step4_causal_rescue import evaluate_rescue
from invariance.train_sequence import evaluate,load_blocks

def save_csv(path,rows):
    if not rows:return
    with open(path,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bases_file",required=True)
    ap.add_argument("--anchor_checkpoint",required=True)
    ap.add_argument("--adapted_checkpoint",required=True)
    ap.add_argument("--data_dir",default="invariance_data/wiki")
    ap.add_argument("--old_language",default="en")
    ap.add_argument("--new_language",default="zh")
    ap.add_argument("--eval_max_blocks",type=int,default=16)
    ap.add_argument("--eval_batch",type=int,default=2)
    ap.add_argument("--alphas",nargs="+",type=float,default=[0.5,1.0])
    ap.add_argument("--limit_draws",type=int,default=2)
    ap.add_argument("--out_dir",required=True)
    args=ap.parse_args()
    if not torch.cuda.is_available():raise RuntimeError("CUDA required")
    if any(x<=0 or x>1 for x in args.alphas):ap.error("alphas must be (0,1]")
    if args.eval_max_blocks<=0:ap.error("Use explicit eval_max_blocks for reproducibility")
    data=torch.load(args.bases_file,map_location="cpu",weights_only=False)
    meta=data["metadata"];bases=data["bases"];layer=int(meta["layer"])
    n=args.limit_draws
    names=[k for k in bases if k.startswith("pc") or any(k.endswith(f"_{i:02d}") for i in range(n))]
    names=sorted(names,key=lambda x:(0 if x.startswith("pc") else 1,x))
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda")
    use_bf16=torch.cuda.is_bf16_supported()
    val={}
    for lang in (args.old_language,args.new_language):
        val[lang]=load_blocks(Path(args.data_dir)/f"{lang}_val.pt")[:args.eval_max_blocks]
        if len(val[lang])==0:raise RuntimeError(f"Empty blocks {lang}")
    print("[load] anchor and adapted 7B",flush=True)
    anchor=load_model(args.anchor_checkpoint,device,use_bf16)
    adapted=load_model(args.adapted_checkpoint,device,use_bf16)
    anchor.eval();adapted.eval()
    for m in (anchor,adapted):
        for p in m.parameters():p.requires_grad_(False)
    base={}
    for lang,blocks in val.items():
        al=evaluate(anchor,blocks,args.eval_batch,device,use_bf16)
        bl=evaluate(adapted,blocks,args.eval_batch,device,use_bf16)
        base[lang]=dict(anchor_loss=al,adapted_loss=bl,forgetting_delta=bl-al)
        print(f"[baseline] {lang}: anchor={al:.6f} adapted={bl:.6f} gap={bl-al:+.6f}",flush=True)
    baseline_path=out/"baselines.json";baseline_path.write_text(json.dumps(base,indent=2))
    rows=[]
    for lang,blocks in val.items():
        energies={};qs={}
        # Natural intervention also yields precise token-level projected energy.
        for name in names:
            q=bases[name];qs[name]=q
            loss,energy=evaluate_rescue(adapted,anchor,blocks,args.eval_batch,device,use_bf16,
                                        layer_no=layer,basis=q,alpha=1.,scale=1.)
            energies[name]=(loss,energy)
            print(f"[natural] {lang:2s} {name:26s} loss_change={loss-base[lang]['adapted_loss']:+.6f} "
                  f"projected_sq_fraction={energy:.7f}",flush=True)
        positive=[energy for _,energy in energies.values() if energy>1e-20]
        if len(positive)!=len(names):raise RuntimeError("Zero projected drift energy")
        target=min(positive)
        for name in names:
            nat_loss,nat_energy=energies[name]
            s=math.sqrt(target/nat_energy)
            for alpha in args.alphas:
                # natural at alpha=1 already computed; no duplicate model eval
                if abs(alpha-1.)<1e-10:natural_loss=nat_loss
                else:
                    natural_loss,_=evaluate_rescue(adapted,anchor,blocks,args.eval_batch,device,use_bf16,
                       layer_no=layer,basis=qs[name],alpha=alpha,scale=1.)
                # equal-energy only scales downward
                if abs(s-1)<1e-8:matched_loss=natural_loss;measured=alpha*alpha*nat_energy
                else:
                    matched_loss,measured=evaluate_rescue(adapted,anchor,blocks,args.eval_batch,device,use_bf16,
                      layer_no=layer,basis=qs[name],alpha=alpha,scale=s)
                row=dict(language=lang,subspace=name,layer=layer,rank=meta["rank"],
                    alpha=alpha,baseline_anchor_loss=base[lang]["anchor_loss"],
                    baseline_adapted_loss=base[lang]["adapted_loss"],
                    forgetting_gap=base[lang]["forgetting_delta"],
                    natural_energy_fraction=nat_energy,target_energy_fraction=target,
                    shrink_scale=s,natural_loss_change=natural_loss-base[lang]["adapted_loss"],
                    matched_loss_change=matched_loss-base[lang]["adapted_loss"],
                    matched_energy_fraction=measured,
                    old_recovery=(float(-(matched_loss-base[lang]["adapted_loss"])/(base[lang]["adapted_loss"]-base[lang]["anchor_loss"]))
                                  if lang==args.old_language and base[lang]["adapted_loss"]>base[lang]["anchor_loss"] else ""),
                    new_gain_cost=(float((matched_loss-base[lang]["adapted_loss"])/(base[lang]["anchor_loss"]-base[lang]["adapted_loss"]))
                                  if lang==args.new_language and base[lang]["anchor_loss"]>base[lang]["adapted_loss"] else ""))
                rows.append(row)
                save_csv(out/"spectrum_rescue_summary.csv",rows)
                print(f"[matched] {lang} {name} alpha={alpha} scale={s:.4f} "
                  f"dLoss={row['matched_loss_change']:+.6f} energy={measured:.7f}",flush=True)
    protocol=dict(anchor_checkpoint=args.anchor_checkpoint,adapted_checkpoint=args.adapted_checkpoint,
        fit_metadata=meta,subspaces=names,eval_max_blocks=args.eval_max_blocks,
        eval_batch=args.eval_batch,alphas=args.alphas,layer=layer,
        matching="per-language global total projected displacement squared fraction; scale down only",
        note="This is Step 6 RESCUE control only, not full removal; no token-level bootstrap CI.",
        limitation="Pooled XNLI drift PCA vs all-token Wiki intervention; fitted PCs are not exactly the all-token drift eigenvectors.")
    (out/"protocol.json").write_text(json.dumps(protocol,indent=2))
    print(f"[DONE] {out/'spectrum_rescue_summary.csv'}",flush=True)
if __name__=="__main__":main()
