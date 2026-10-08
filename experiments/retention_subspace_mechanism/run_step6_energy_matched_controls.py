#!/usr/bin/env python3
import argparse, csv, json, math, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.random_layer_freeze_pilot.run_pilot import load_model
from experiments.retention_subspace_mechanism.run_step3_causal_removal import evaluate_removed
from experiments.retention_subspace_mechanism.run_step4_causal_rescue import evaluate_rescue
from experiments.retention_subspace_mechanism.subspace_extractors import orthonormal_random
from invariance.train_sequence import evaluate, load_blocks

def mean(xs): return sum(xs)/len(xs)
def std(xs):
    if len(xs) < 2: return 0.0
    m=mean(xs)
    return math.sqrt(sum((x-m)**2 for x in xs)/(len(xs)-1))

def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

def main():
    ap=argparse.ArgumentParser(description="Step 6: energy-matched multi-random controls for causal removal/rescue.")
    ap.add_argument("--anchor_checkpoint", default="invariance_runs/sequence_seed0/en__zh/stage1_en")
    ap.add_argument("--adapted_checkpoint", default="mechanism_runs/step1_layer_lambda_sweep/checkpoints/full_ft")
    ap.add_argument("--subspace_file", default="mechanism_runs/step2_layer20_subspaces_v2/layer20_subspaces.pt")
    ap.add_argument("--data_dir", default="invariance_data/wiki")
    ap.add_argument("--old_language", default="en")
    ap.add_argument("--new_language", default="zh")
    ap.add_argument("--subspaces", nargs="+", default=["transfer","drift","isr_cov","isr_multiclass","vicreg"])
    ap.add_argument("--strengths", type=float, nargs="+", default=[0.25,0.5,1.0])
    ap.add_argument("--n_random", type=int, default=8)
    ap.add_argument("--random_seed", type=int, default=7300)
    ap.add_argument("--eval_max_blocks", type=int, default=128)
    ap.add_argument("--eval_batch", type=int, default=8)
    ap.add_argument("--max_scale", type=float, default=8.0)
    ap.add_argument("--out_dir", default="mechanism_runs/step6_energy_matched_controls")
    args=ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU required")
    if args.n_random < 2: raise ValueError("--n_random must be >= 2")

    device=torch.device("cuda")
    use_bf16=torch.cuda.is_bf16_supported()
    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    payload=torch.load(args.subspace_file, map_location="cpu")
    layer=int(payload["layer"]); center=payload["center"].float()
    subspaces={k:v.float() for k,v in payload["subspaces"].items()}
    dim=int(payload["hidden_dim"])
    for s in args.subspaces:
        if s not in subspaces: raise KeyError(f"Missing subspace: {s}")

    val={}
    for lang in [args.old_language,args.new_language]:
        b=load_blocks(Path(args.data_dir)/f"{lang}_val.pt")
        if args.eval_max_blocks>0: b=b[:args.eval_max_blocks]
        val[lang]=b

    print("Loading models...")
    anchor=load_model(args.anchor_checkpoint, device, use_bf16)
    adapted=load_model(args.adapted_checkpoint, device, use_bf16)
    anchor.eval(); adapted.eval()
    for p in anchor.parameters(): p.requires_grad_(False)
    for p in adapted.parameters(): p.requires_grad_(False)

    base_anchor={}; base_adapted={}
    for lang in val:
        base_anchor[lang]=evaluate(anchor,val[lang],args.eval_batch,device,use_bf16)
        base_adapted[lang]=evaluate(adapted,val[lang],args.eval_batch,device,use_bf16)
        print(f"[baseline] {lang} anchor={base_anchor[lang]:.6f} adapted={base_adapted[lang]:.6f}")

    removal_rows=[]; rescue_rows=[]
    for si,real in enumerate(args.subspaces):
        q_real=subspaces[real]
        rank=int(q_real.shape[1])
        randoms=[
            (f"energy_random_{real}_{j:02d}", orthonormal_random(dim,rank,args.random_seed+1000*si+j))
            for j in range(args.n_random)
        ]
        for lang in val:
            # Calibrate unscaled intervention energy at strength=1.
            _, real_rem_frac=evaluate_removed(
                adapted,val[lang],args.eval_batch,device,use_bf16,
                layer_no=layer,basis=q_real,center=center,strength=1.0,scale=1.0)
            _, real_res_frac=evaluate_rescue(
                adapted,anchor,val[lang],args.eval_batch,device,use_bf16,
                layer_no=layer,basis=q_real,alpha=1.0,scale=1.0)

            cal=[]
            for rn,rq in randoms:
                _, rf=evaluate_removed(
                    adapted,val[lang],args.eval_batch,device,use_bf16,
                    layer_no=layer,basis=rq,center=center,strength=1.0,scale=1.0)
                _, sf=evaluate_rescue(
                    adapted,anchor,val[lang],args.eval_batch,device,use_bf16,
                    layer_no=layer,basis=rq,alpha=1.0,scale=1.0)
                rem_scale=min(math.sqrt(real_rem_frac/max(rf,1e-30)),args.max_scale)
                res_scale=min(math.sqrt(real_res_frac/max(sf,1e-30)),args.max_scale)
                cal.append((rn,rq,rem_scale,res_scale,rf,sf))

            for strength in args.strengths:
                real_loss, real_frac=evaluate_removed(
                    adapted,val[lang],args.eval_batch,device,use_bf16,
                    layer_no=layer,basis=q_real,center=center,strength=strength,scale=1.0)
                real_delta=real_loss-base_adapted[lang]
                random_deltas=[]
                for rn,rq,rem_scale,res_scale,rf,sf in cal:
                    loss,frac=evaluate_removed(
                        adapted,val[lang],args.eval_batch,device,use_bf16,
                        layer_no=layer,basis=rq,center=center,strength=strength,scale=rem_scale)
                    delta=loss-base_adapted[lang]; random_deltas.append(delta)
                    removal_rows.append({
                        "real_subspace":real,"control":rn,"language":lang,"strength":strength,
                        "rank":rank,"real_loss_delta":real_delta,"random_loss_delta":delta,
                        "excess_vs_energy_matched_random":real_delta-delta,
                        "real_perturbation_fraction":real_frac,"random_perturbation_fraction":frac,
                        "random_scale":rem_scale})
                print(f"[remove] {real:16s} {lang} beta={strength:.2f} real={real_delta:+.6f} random={mean(random_deltas):+.6f}±{std(random_deltas):.6f}")

                real_loss, real_frac=evaluate_rescue(
                    adapted,anchor,val[lang],args.eval_batch,device,use_bf16,
                    layer_no=layer,basis=q_real,alpha=strength,scale=1.0)
                real_delta=real_loss-base_adapted[lang]
                random_deltas=[]
                for rn,rq,rem_scale,res_scale,rf,sf in cal:
                    loss,frac=evaluate_rescue(
                        adapted,anchor,val[lang],args.eval_batch,device,use_bf16,
                        layer_no=layer,basis=rq,alpha=strength,scale=res_scale)
                    delta=loss-base_adapted[lang]; random_deltas.append(delta)
                    rescue_rows.append({
                        "real_subspace":real,"control":rn,"language":lang,"alpha":strength,
                        "rank":rank,"real_loss_change":real_delta,"random_loss_change":delta,
                        "real_perturbation_fraction":real_frac,"random_perturbation_fraction":frac,
                        "random_scale":res_scale})
                print(f"[rescue] {real:16s} {lang} alpha={strength:.2f} real={real_delta:+.6f} random={mean(random_deltas):+.6f}±{std(random_deltas):.6f}")

    write_csv(out/"energy_matched_removal_draws.csv",removal_rows)
    write_csv(out/"energy_matched_rescue_draws.csv",rescue_rows)

    summary=[]
    for real in args.subspaces:
        for strength in args.strengths:
            for lang in val:
                rr=[r for r in removal_rows if r["real_subspace"]==real and r["strength"]==strength and r["language"]==lang]
                rs=[r for r in rescue_rows if r["real_subspace"]==real and r["alpha"]==strength and r["language"]==lang]
                rem_rand=[r["random_loss_delta"] for r in rr]
                res_rand=[r["random_loss_change"] for r in rs]
                summary.append({
                    "real_subspace":real,"strength_or_alpha":strength,"language":lang,"n_random":len(rr),
                    "removal_real_delta":rr[0]["real_loss_delta"],
                    "removal_random_mean":mean(rem_rand),"removal_random_std":std(rem_rand),
                    "removal_excess_vs_random_mean":rr[0]["real_loss_delta"]-mean(rem_rand),
                    "rescue_real_loss_change":rs[0]["real_loss_change"],
                    "rescue_random_mean_loss_change":mean(res_rand),"rescue_random_std_loss_change":std(res_rand),
                    "rescue_real_minus_random_mean":rs[0]["real_loss_change"]-mean(res_rand),
                    "random_rescue_at_least_as_good_fraction":sum(x<=rs[0]["real_loss_change"] for x in res_rand)/len(res_rand),
                })
    write_csv(out/"energy_matched_summary.csv",summary)
    (out/"manifest.json").write_text(json.dumps({
        "layer":layer,"subspaces":args.subspaces,"n_random":args.n_random,
        "random_seed":args.random_seed,"strengths":args.strengths,
        "control_definition":"Rank-matched isotropic random bases globally rescaled per language so intervention squared norm matches the corresponding real-subspace perturbation.",
        "note":"Scaled random controls are energy-matched perturbations, not literal component removals when scale > 1."
    },indent=2))
    print(f"Saved: {out/'energy_matched_summary.csv'}")

if __name__=="__main__":
    main()
