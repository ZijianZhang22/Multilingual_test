#!/usr/bin/env python3
"""Unify Drift/Transfer/ISR semantic patching, bidirectional LM rescue and probe.

Results are keyed by subspace, not claimed causal chains. Row metrics from
distinct model/tasks/splits are shown side-by-side with provenance.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from experiments.literature_measurements.core import save_csv
from experiments.literature_measurements.subspace_core import REAL_SPACES


def read_csv(path):
    if not path:
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def avg(items, key):
    vals = [float(row[key]) for row in items if row.get(key) not in ("",None)]
    return sum(vals)/len(vals) if vals else ""


def unified(semantic, bidir, probe, old_language, new_language):
    out = []
    for name in REAL_SPACES:
        sr = [r for r in semantic if r.get("subspace")==name
              and r.get("energy_mode")=="matched"]
        tr = [r for r in sr if r.get("donor_condition")=="same_pair_cross_lang"]
        cr = [r for r in sr if r.get("donor_condition")=="different_pair_cross_lang_same_label"]
        br = [r for r in bidir if r.get("subspace")==name and
              r.get("language")==old_language]
        restore = [r for r in br if r.get("direction")=="restore"]
        induce = [r for r in br if r.get("direction")=="induce"]
        nr = [r for r in bidir if r.get("subspace")==name and
              r.get("language")==new_language and r.get("direction")=="restore"]
        pr = [r for r in probe if r.get("basis")==name and
              r.get("checkpoint")=="adapted" and r.get("task") in ("direct","loo")]
        row = {
            "subspace": name,
            "semantic_same_pair_gold_nll_delta": avg(tr, "gold_nll_delta"),
            "semantic_same_label_control_gold_nll_delta": avg(cr,"gold_nll_delta"),
            "semantic_same_pair_minus_control": (
                avg(tr,"gold_nll_delta") - avg(cr,"gold_nll_delta")
                if tr and cr else ""),
            "semantic_n_targets": (tr[0].get("n") if tr else 0),
            "old_restore_nll_delta": avg(restore,"loss_delta"),
            "old_restore_gap_fraction": avg(restore,"recovery_fraction"),
            "old_induce_nll_delta": avg(induce,"loss_delta"),
            "old_induced_forgetting_fraction": avg(induce,"induced_forgetting_fraction"),
            "new_language_restore_nll_delta": avg(nr,"loss_delta"),
            "transfer_probe_accuracy": avg([r for r in pr if r.get("task")=="direct"], "accuracy"),
            "loo_probe_accuracy": avg([r for r in pr if r.get("task")=="loo"], "accuracy"),
            "semantic_available": bool(tr),
            "bidir_available": bool(restore and induce),
            "probe_available": bool(pr),
        }
        out.append(row)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--semantic_csv", default="")
    p.add_argument("--bidirectional_csv", default="")
    p.add_argument("--probe_csv", default="")
    p.add_argument("--old_language", default="en")
    p.add_argument("--new_language", default="zh")
    p.add_argument("--out_dir", required=True)
    a = p.parse_args()
    semantic, bidir, probe = (read_csv(a.semantic_csv),
                               read_csv(a.bidirectional_csv),
                               read_csv(a.probe_csv))
    if not any((semantic,bidir,probe)):
        p.error("At least one input CSV needs data")
    names = {r.get("subspace") for r in semantic+bidir}
    names |= {r.get("basis") for r in probe}
    unknown = {s for s in names if s and s not in REAL_SPACES
               and not s.startswith("random_") and s not in
               ("anchor_pca","drift_train","random")}
    if unknown:
        print(f"Ignoring extra legacy spaces in auxiliary inputs: {sorted(unknown)}")
    rows = unified(semantic,bidir,probe,a.old_language,a.new_language)
    out = Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    save_csv(out/"subspace_functional_summary.csv",rows)
    (out/"summary_sources.json").write_text(json.dumps({
        **vars(a),
        "warning":"Descriptive linkage only; task prompt and Wiki token distributions differ. "
                  "Do not treat correlations or directionality as training-mechanism proof.",
        "spaces": list(REAL_SPACES)
    },indent=2))
    print(f"Saved {out/'subspace_functional_summary.csv'}")


if __name__=="__main__":
    main()
