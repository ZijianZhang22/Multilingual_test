#!/usr/bin/env python3
"""Build paper-main replication subspaces: Transfer, Drift, ISR-Cov, ISR-Multiclass, VICReg."""
import argparse, json, subprocess, sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
INV=ROOT/"invariance"
from experiments.retention_subspace_mechanism.subspace_extractors import (
    fit_isr_cov, fit_isr_multiclass_semantic, fit_vicreg_linear, orthonormal_random
)

def run(cmd):
    cmd=[str(x) for x in cmd]
    print("\n>>>"," ".join(cmd),flush=True)
    subprocess.run(cmd,check=True)

def maybe(path,cmd,force=False):
    path=Path(path)
    if path.exists() and not force:
        print(f"SKIP existing: {path}",flush=True); return
    run(cmd)

def read_jsonl(path):
    rows=[]
    with Path(path).open("r",encoding="utf-8") as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows

def fit_drift_basis(anchor_features,adapted_features,layer,rank):
    a=torch.load(anchor_features,map_location="cpu")
    b=torch.load(adapted_features,map_location="cpu")
    xa=a["features"][str(layer)].float(); xb=b["features"][str(layer)].float()
    if xa.shape!=xb.shape or a["example_ids"]!=b["example_ids"]:
        raise ValueError("Anchor/adapted probe features are not aligned.")
    delta=xb-xa; dc=delta-delta.mean(dim=0,keepdim=True)
    q=min(rank,dc.shape[0],dc.shape[1])
    _,s,v=torch.pca_lowrank(dc,q=q,center=False)
    basis=v[:,:q]
    ev=s[:q].pow(2); ev=ev/ev.sum().clamp_min(1e-12)
    return basis,delta,ev

def overlap_stats(qa,qb):
    s=torch.linalg.svdvals(qa.T@qb)
    return {"mean_squared_cosine":float((s**2).mean()),"max_cosine":float(s.max())}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--anchor_checkpoint",required=True)
    ap.add_argument("--adapted_checkpoint",required=True)
    ap.add_argument("--languages",nargs="+",default=["en","zh","fr","de","es"])
    ap.add_argument("--layer",type=int,required=True); ap.add_argument("--rank",type=int,default=64)
    ap.add_argument("--inlp_iters",type=int,default=32)
    ap.add_argument("--probe_train_per_lang",type=int,default=1200); ap.add_argument("--probe_test_per_lang",type=int,default=1200)
    ap.add_argument("--aligned_examples",type=int,default=2000); ap.add_argument("--extract_batch",type=int,default=4)
    ap.add_argument("--isr_cov_class",type=int,default=0)
    ap.add_argument("--vicreg_epochs",type=int,default=300)
    ap.add_argument("--vicreg_lr",type=float,default=3e-2)
    ap.add_argument("--out_dir",required=True); ap.add_argument("--force",action="store_true")
    args=ap.parse_args()
    out=Path(args.out_dir); data_dir=out/"data"; feat_dir=out/"features"
    data_dir.mkdir(parents=True,exist_ok=True); feat_dir.mkdir(parents=True,exist_ok=True)
    probe=data_dir/"xnli_probe.jsonl"; aligned=data_dir/"xnli_aligned.jsonl"
    anchor_probe=feat_dir/"anchor_probe.pt"; adapted_probe=feat_dir/"adapted_probe.pt"; anchor_aligned=feat_dir/"anchor_aligned.pt"
    lang_file=out/"language_inlp.pt"; transfer_file=out/f"transfer_rank{args.rank}.pt"
    maybe(probe,[sys.executable,INV/"prepare_xnli.py","--languages",*args.languages,"--train_per_lang",args.probe_train_per_lang,"--test_per_lang",args.probe_test_per_lang,"--seed",2026,"--out_file",probe],args.force)
    maybe(aligned,[sys.executable,INV/"prepare_aligned_xnli.py","--languages",*args.languages,"--split","validation","--n_examples",args.aligned_examples,"--seed",2026,"--out_file",aligned],args.force)
    for ckpt,data_file,out_file in [(args.anchor_checkpoint,probe,anchor_probe),(args.adapted_checkpoint,probe,adapted_probe),(args.anchor_checkpoint,aligned,anchor_aligned)]:
        maybe(out_file,[sys.executable,INV/"extract_hidden.py","--checkpoint",ckpt,"--data_file",data_file,"--out_file",out_file,"--layers",args.layer,"--batch_size",args.extract_batch],args.force)
    maybe(lang_file,[sys.executable,INV/"fit_inlp_language_subspace.py","--features_file",anchor_probe,"--out_file",lang_file,"--layer",args.layer,"--iters",args.inlp_iters,"--seed",0],args.force)
    maybe(transfer_file,[sys.executable,INV/"fit_transferable_subspace.py","--features_file",anchor_aligned,"--aligned_data_file",aligned,"--language_subspace_file",lang_file,"--out_file",transfer_file,"--layer",args.layer,"--rank",args.rank],args.force)
    probe_payload=torch.load(anchor_probe,map_location="cpu")
    x_probe=probe_payload["features"][str(args.layer)].float(); center=x_probe.mean(dim=0)
    q_transfer=torch.load(transfer_file,map_location="cpu")["transferable_subspace_basis"].float()
    q_drift,delta,drift_ev=fit_drift_basis(anchor_probe,adapted_probe,args.layer,args.rank)
    train_idx=torch.tensor([i for i,s in enumerate(probe_payload["splits"]) if s=="probe_train"],dtype=torch.long)
    x_train=x_probe[train_idx]
    langs_train=[probe_payload["languages"][i] for i in train_idx.tolist()]
    labels_train=probe_payload["labels"][train_idx].long()
    q_isr_cov,isr_cov_meta=fit_isr_cov(x_train,langs_train,labels_train,rank=args.rank,class_label=args.isr_cov_class)
    q_isr_multi,q_isr_spurious,isr_multi_meta=fit_isr_multiclass_semantic(x_train,langs_train,labels_train,rank=args.rank)
    aligned_payload=torch.load(anchor_aligned,map_location="cpu")
    x_aligned=aligned_payload["features"][str(args.layer)].float()
    aligned_rows=read_jsonl(aligned)
    pair_ids=[r["pair_id"] for r in aligned_rows]
    if len(pair_ids)!=x_aligned.shape[0]:
        raise ValueError("Aligned JSONL/features length mismatch for VICReg.")
    q_vicreg,vicreg_meta=fit_vicreg_linear(
        x_aligned,pair_ids,rank=args.rank,epochs=args.vicreg_epochs,
        lr=args.vicreg_lr,seed=0
    )
    real={"transfer":q_transfer,"drift":q_drift,"isr_cov":q_isr_cov,"isr_multiclass":q_isr_multi,"vicreg":q_vicreg}
    subspaces={}; controls={}; dim=x_probe.shape[1]
    for i,(name,q) in enumerate(real.items()):
        subspaces[name]=q; rn=f"random_{name}"
        subspaces[rn]=orthonormal_random(dim,q.shape[1],seed=4101+i); controls[name]=rn
    overlaps={}
    names=list(real)
    for i,a in enumerate(names):
        for b in names[i:]:
            overlaps[f"{a}__{b}"]=overlap_stats(real[a],real[b])
    artifact={
        "layer":args.layer,"hidden_dim":dim,"anchor_checkpoint":args.anchor_checkpoint,"adapted_checkpoint":args.adapted_checkpoint,
        "fit_languages":args.languages,"center":center,"subspaces":subspaces,"real_subspaces":names,
        "matched_random_controls":controls,"ranks":{k:int(v.shape[1]) for k,v in subspaces.items()},
        "drift_mean_delta_l2":float(delta.norm(dim=1).mean()),"drift_explained_fraction_within_rank":drift_ev,
        "isr_cov_metadata":isr_cov_meta,"isr_multiclass_metadata":isr_multi_meta,
        "isr_multiclass_spurious_basis":q_isr_spurious,"vicreg_metadata":vicreg_meta,"overlaps":overlaps
    }
    out_file=out/"core_subspaces.pt"; torch.save(artifact,out_file)
    summary={"layer":args.layer,"hidden_dim":dim,"ranks":artifact["ranks"],"drift_mean_delta_l2":artifact["drift_mean_delta_l2"],"overlaps":overlaps,"isr_cov_metadata":isr_cov_meta,"isr_multiclass_metadata":isr_multi_meta,"vicreg_metadata":vicreg_meta}
    (out/"summary.json").write_text(json.dumps(summary,indent=2))
    print(f"Saved: {out_file}",flush=True)
if __name__=="__main__": main()
