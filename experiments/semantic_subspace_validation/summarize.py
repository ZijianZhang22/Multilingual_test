#!/usr/bin/env python3
"""Summarize semantic validation. ISR-derived invariance is not proven semantics."""
import argparse,csv,json
from pathlib import Path
from shared import write_json

def rows(path):
    p=Path(path)
    if not p.is_file():return []
    with p.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run_dir',required=True)
    ap.add_argument('--out_dir',required=True)
    a=ap.parse_args()
    run,out=Path(a.run_dir),Path(a.out_dir)
    p=rows(out/'probes_and_retrieval/probe_results.csv')
    r=rows(out/'probes_and_retrieval/retrieval_results.csv')
    c=rows(out/'causal_semantics/causal_summary.csv')
    s=rows(run/'drift_isr_partition/partition_rescue_summary.csv')
    bases=['top16','bottom16','top32','bottom32','isr_multiclass','transfer','vicreg','drift','pca64','random16','random32','random64']
    def probe(task):
        return {b:float(next(row['accuracy'] for row in p if row['subspace']==b and row['task']==task))
                for b in bases if any(row['subspace']==b and row['task']==task for row in p)}
    nli,lid=probe('nli'),probe('language_id')
    causal={b:{t:next((row for row in c if row['subspace']==b and row['task']==t and row['variant']=='norm_matched'),None)
               for t in ('nli','language_id')} for b in bases}
    report=dict(status='NOT PROVEN: ISR invariance is not synonymous with semantics',
        probe_nli=nli,probe_language_id=lid,retrieval=r,causal=causal,step7=s,
        limitations=['XNLI NLI is not full semantics',
                     'Probe test labels held out, but drift subspace may include unlabeled test inputs',
                     'Mean-pooling for subspace fit versus last-token behavioral intervention',
                     'Small causal sample (default 18), not a statistically conclusive test',
                     'Random baselines and matching do not rule out all intervention artifacts'])
    write_json(out/'evidence_report.json',report)
    lines=['# Semantic subspace validation','',
           '**ISR-aligned is not proven semantic.**','','## Probe accuracy','',
           '| Basis | NLI | Language ID |','|---|---:|---:|']
    for b in bases:
        if b in nli or b in lid:
            lines.append(f'| {b} | {nli.get(b,float("nan")):.4f} | {lid.get(b,float("nan")):.4f} |')
    lines+=['','## Causal NLI (norm matched)','',
            '| Basis | Baseline acc | Acc change | NLL increase |',
            '|---|---:|---:|---:|']
    for b in bases:
        item=causal[b]['nli']
        if item:
            lines.append(f'| {b} | {float(item["baseline_accuracy"]):.3f} | {float(item["accuracy_change"]):+.3f} | {float(item["nll_increase"]):+.4f} |')
    baseline=next((float(row['baseline_accuracy']) for row in c if row['task']=='nli'),None)
    if baseline is not None and baseline<.45:
        lines+=['','**WARNING:** Near-chance zero-shot NLI baseline; avoid semantic causal interpretation.']
    lines+=['','## Retrieval','',f'{len(r)} language/subspace rows in retrieval_results.csv.','','## Original Step 7','',f'{len(s)} rows; inspect original energy-matched and natural rescue columns.','',
            '## Interpretation caveats','']+['- '+x for x in report['limitations']]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Saved:',out/'REPORT.md',out/'evidence_report.json',flush=True)

if __name__=='__main__':main()
