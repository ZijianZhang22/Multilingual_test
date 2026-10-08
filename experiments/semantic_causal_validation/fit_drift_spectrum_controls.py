#!/usr/bin/env python3
"""Fit a rank-256 adaptation-drift PCA and matched-rank spectral controls.

Uses pre-extracted aligned anchor/adapted XNLI hidden states. Only probe_train
examples are used by default, so held-out XNLI probe_test is not used for fit.
For Wiki validation, XNLI features and Wiki eval are also distinct corpora.

NOTE: "last" features are last unprompted XNLI tokens; the subsequent Step-6
causal rescue edits all positions in a Wiki LM block. Test mean-pool separately.
"""
import argparse
import json
import math
from pathlib import Path

import torch


def ortho(x):
    return torch.linalg.qr(x.float(), mode="reduced")[0]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--anchor_features",required=True)
    ap.add_argument("--adapted_features",required=True)
    ap.add_argument("--layer",type=int,default=23)
    ap.add_argument("--num_pcs",type=int,default=256)
    ap.add_argument("--rank",type=int,default=64)
    ap.add_argument("--draws",type=int,default=4)
    ap.add_argument("--fit_split",default="probe_train")
    ap.add_argument("--seed",type=int,default=2026)
    ap.add_argument("--niter",type=int,default=3)
    ap.add_argument("--out_file",required=True)
    args=ap.parse_args()
    if args.num_pcs<2*args.rank or args.num_pcs%args.rank:
        ap.error("--num_pcs must be multiple of --rank and >= 2*rank")
    if args.draws<1:ap.error("--draws must be >=1")

    a=torch.load(args.anchor_features,map_location="cpu",weights_only=False)
    b=torch.load(args.adapted_features,map_location="cpu",weights_only=False)
    key=str(args.layer)
    if a["example_ids"]!=b["example_ids"]:
        raise ValueError("Features are not example-id aligned")
    if a.get("pool")!=b.get("pool"):
        raise ValueError("Anchor/adapted pooling mismatch")
    if key not in a["features"] or key not in b["features"]:
        raise ValueError(f"Missing layer {key}")
    indices=[i for i,s in enumerate(a["splits"]) if s==args.fit_split]
    if len(indices)<=args.num_pcs+16:raise ValueError("Insufficient fitting examples")
    print(f"[fit] pool={a.get('pool')} n_fit={len(indices)} layer={args.layer}",flush=True)
    xa=a["features"][key][indices].float()
    xb=b["features"][key][indices].float()
    if xa.shape!=xb.shape:raise ValueError("Feature shapes differ")
    delta=xb-xa
    del xa,xb
    delta=delta-delta.mean(dim=0,keepdim=True)
    d=int(delta.shape[1])
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matrix=delta.to(dev)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    q=min(args.num_pcs+16,min(matrix.shape))
    print(f"[fit] randomized PCA q={q} iters={args.niter} device={dev}",flush=True)
    _,sing,v=torch.pca_lowrank(matrix,q=q,center=False,niter=args.niter)
    components=ortho(v[:,:args.num_pcs].cpu())
    # QR maintains span and numerical orthonormality; q from PCA is already orthonormal.
    # QR sign flips are irrelevant to projection. Importance order is unchanged.
    sing=sing[:args.num_pcs].cpu()
    total_energy=float(matrix.square().sum())
    cumulative=float((sing.square().sum()/max(total_energy,1e-30)).item())
    print(f"[fit] leading {args.num_pcs} PCs capture {cumulative:.4%} train delta variance",flush=True)
    del matrix,delta
    gen=torch.Generator(device="cpu").manual_seed(args.seed+159)
    r=args.rank;N=args.num_pcs
    controls={}
    for i in range(N//r):
        controls[f"pc{i*r+1:03d}_{(i+1)*r:03d}"]=components[:,i*r:(i+1)*r].clone()
    for i in range(args.draws):
        ind=torch.randperm(N,generator=gen)[:r]
        controls[f"top{N}_subset_{i:02d}"]=components[:,ind]
        ind_tail=r+torch.randperm(N-r,generator=gen)[:r]
        controls[f"tail{r+1}_{N}_subset_{i:02d}"]=components[:,ind_tail]
        mix=ortho(torch.randn(N,r,generator=gen))
        controls[f"top{N}_haar_{i:02d}"]=components@mix
        controls[f"isotropic_{i:02d}"]=ortho(torch.randn(d,r,generator=gen))
    check={k:float((v.T@v-torch.eye(r)).abs().max()) for k,v in controls.items()}
    if max(check.values())>2e-4:raise RuntimeError(f"Orthonormality failure: {max(check.values())}")
    output=Path(args.out_file);output.parent.mkdir(parents=True,exist_ok=True)
    meta=dict(layer=args.layer,pool=a.get("pool"),rank=r,num_pcs=N,n_examples=len(indices),
        fit_split=args.fit_split,seed=args.seed,niter=args.niter,
        anchor_features=args.anchor_features,adapted_features=args.adapted_features,
        variance_fraction_top_pcs=cumulative,
        method="randomized PCA of centered adapted-minus-anchor features; seed fixed; q=num_pcs+16",
        comparison="pc001_064 against PC bands and within-spectrum random controls",
        leakage_note="XNLI probe_train features only; Wiki validation held separately",
        intervention_mismatch="PCA fitted on pooled XNLI features, while Step-6 rescue edits all LM token positions.")
    torch.save({"bases":controls,"singular_values":sing,"metadata":meta},output)
    output.with_suffix(".json").write_text(json.dumps(dict(meta,controls=list(controls),
        max_orthogonality_error=max(check.values())),indent=2))
    print(f"[done] saved {len(controls)} rank-{r} bases to {output}",flush=True)
    for i in range(N//r):
        s=sing[i*r:(i+1)*r]
        print(f"    PC {i*r+1}-{(i+1)*r}: fitting-energy-share {float(s.square().sum()/total_energy):.5f}",flush=True)

if __name__=="__main__":main()
