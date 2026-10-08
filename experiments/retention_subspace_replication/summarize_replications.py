#!/usr/bin/env python3
import argparse, csv, json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",default="replication_runs/qwen_scale_seed"); args=ap.parse_args()
    root=Path(args.root); rows=[]
    for manifest_path in sorted(root.glob("*/seed*/replication_manifest.json")):
        run_dir=manifest_path.parent
        train_path=run_dir/"training/training_summary.json"
        energy_path=run_dir/"energy_controls/energy_matched_summary.csv"
        if not train_path.exists() or not energy_path.exists(): continue
        meta=json.loads(manifest_path.read_text()); train=json.loads(train_path.read_text())
        forgetting=float(train["forgetting"]); gain=float(train["new_language_gain"])
        with energy_path.open(newline="") as f:
            for r in csv.DictReader(f):
                lang=r["language"]; alpha=float(r["strength_or_alpha"])
                real_change=float(r["rescue_real_loss_change"]); random_change=float(r["rescue_random_mean_loss_change"])
                row={"model_tag":meta["model_tag"],"model_name":meta["model_name"],"seed":meta["seed"],
                     "n_layers":meta["n_layers"],"layer":meta["selected_layer_1based"],"relative_depth":meta["selected_relative_depth"],
                     "subspace":r["real_subspace"],"alpha":alpha,"language":lang,"forgetting":forgetting,
                     "new_language_gain":gain,"real_rescue_loss_change":real_change,
                     "random_rescue_loss_change":random_change,"real_minus_random":real_change-random_change}
                if lang==train["old_language"] and forgetting>0:
                    row["real_old_recovery_fraction"]=-real_change/forgetting
                    row["random_old_recovery_fraction"]=-random_change/forgetting
                    row["recovery_excess_fraction"]=(-real_change+random_change)/forgetting
                else:
                    row["real_old_recovery_fraction"]=""; row["random_old_recovery_fraction"]=""; row["recovery_excess_fraction"]=""
                rows.append(row)
    if not rows:
        print("No completed replication runs found."); return
    out=root/"replication_summary.csv"
    with out.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"Saved: {out}")
    print("\n=== DRIFT old-language rescue ===")
    for r in rows:
        if r["subspace"]=="drift" and r["language"]=="en":
            print(f"{r['model_tag']:4s} seed={r['seed']} layer={r['layer']} alpha={r['alpha']:.2f} real={r['real_old_recovery_fraction']:+.3f} random={r['random_old_recovery_fraction']:+.3f} excess={r['recovery_excess_fraction']:+.3f}")
if __name__=="__main__": main()
