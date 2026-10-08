#!/usr/bin/env python3
"""Frozen-model NLI/language causal ablation and activation swapping.

Interpret cautiously: bases were fitted on mean-pooled states, while
interventions edit only the last prompt token. This is a behavioral test,
not proof of semantic disentanglement.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from shared import read_jsonl, write_csv, write_json

LABEL_LETTERS=[' A',' B',' C']
LANG_LABELS={'en':0,'zh':1,'fr':2}

def make_prompt(row,task):
    prem,hypo=row['premise'],row['hypothesis']
    if task=='nli':
        return ('Read the premise and hypothesis. Decide whether the hypothesis '
                'follows from the premise. Choose exactly one letter: '
                'A = entailment, B = neutral, C = contradiction.\n'
                f'Premise: {prem}\nHypothesis: {hypo}\nAnswer:')
    if task=='language_id':
        return ('Identify the language used in the premise. Choose exactly one '
                'letter: A = English, B = Chinese, C = French.\n'
                f'Premise: {prem}\nAnswer:')
    raise ValueError(task)

def letter_ids(tok):
    result=[tok.encode(x,add_special_tokens=False) for x in LABEL_LETTERS]
    if not all(len(x)==1 for x in result) or len({x[0] for x in result})<3:
        raise RuntimeError(f'Expected three distinct single token continuations; got {result}')
    return [x[0] for x in result]

def model_layers(model):
    if hasattr(model,'model') and hasattr(model.model,'layers'):return model.model.layers
    raise TypeError('Expected Qwen causal LM model.layers')

def replace_hidden(original,altered):
    return (altered,*original[1:]) if isinstance(original,tuple) else altered

def choose_samples(rows,n,seed):
    valid=[r for r in rows if r.get('split')=='probe_test'
           and r['language'] in LANG_LABELS and int(r['label']) in (0,1,2)]
    grouped=defaultdict(list)
    for row in valid:grouped[(row['language'],int(row['label']))].append(row)
    rng=np.random.default_rng(seed)
    per=max(1,n//9); chosen=[]
    for key in sorted(grouped):
        opts=grouped[key]
        chosen.extend([opts[i] for i in rng.permutation(len(opts))[:per]])
    if len(chosen)<6:raise ValueError('Not enough EN/ZH/FR held-out XNLI samples')
    return chosen[:n]

def donor_for(rows,receiver,task):
    lang,label=receiver['language'],int(receiver['label'])
    if task=='nli':
        matches=[r for r in rows if r['language']==lang and int(r['label'])!=label]
    else:
        matches=[r for r in rows if r['language']!=lang and int(r['label'])==label]
    return matches[0] if matches else None

def probs_from_logits(logits,ids):
    return torch.softmax(logits[-1,ids].float(),dim=-1)

def score(probs,gold):
    return dict(accuracy=float(int(probs.argmax().item()==gold)),
                gold_p=float(probs[gold]),
                gold_nll=float(-torch.log(probs[gold].clamp_min(1e-12))))

def load_model(checkpoint,device):
    from transformers import AutoModelForCausalLM,AutoTokenizer
    tok=AutoTokenizer.from_pretrained(checkpoint,use_fast=True)
    model=AutoModelForCausalLM.from_pretrained(
        checkpoint,torch_dtype=torch.bfloat16 if device.type=='cuda'
        and torch.cuda.is_bf16_supported() else torch.float32,
        low_cpu_mem_usage=True,attn_implementation='sdpa').to(device).eval()
    model.config.use_cache=False
    for p in model.parameters():p.requires_grad_(False)
    return model,tok

def run(args):
    torch.manual_seed(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model,tok=load_model(args.checkpoint,device)
    ids=letter_ids(tok)
    payload=torch.load(args.bases_file,map_location='cpu',weights_only=False)
    core=torch.load(args.subspace_file,map_location='cpu',weights_only=False)
    layer=int(payload['layer'])
    basis={name:q.to(device).float() for name,q in payload['bases'].items()
           if name in ('isr_multiclass','isr_cov','vicreg','transfer','drift',
                       'top16','bottom16','top32','bottom32','pca64',
                       'random16','random32','random64') or name.startswith('random32_draw')}
    if args.only:
        requested=set(args.only.split(','))
        basis={name:q for name,q in basis.items() if name in requested}
    center=core['center'].to(device=device,dtype=torch.float32)
    block=model_layers(model)[layer-1]
    samples=choose_samples(read_jsonl(args.probe_data),args.n_examples,args.seed)
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    if args.max_length<128:raise ValueError('--max_length must be >=128')

    def encode(row,task):
        x=tok(make_prompt(row,task),return_tensors='pt',truncation=False)
        if x['input_ids'].shape[1]>args.max_length:
            raise ValueError(f'Prompt {row["example_id"]} exceeds {args.max_length} tokens')
        return {k:v.to(device) for k,v in x.items()}

    def forward(enc,hook_fn=None):
        handle=block.register_forward_hook(hook_fn) if hook_fn else None
        try:
            with torch.inference_mode():
                logits=model(**enc,use_cache=False).logits[0]
                return probs_from_logits(logits,ids).detach().cpu()
        finally:
            if handle is not None:handle.remove()

    def get_last_hidden(enc):
        captured={}
        def capture(_module,_inputs,output):
            h=output[0] if isinstance(output,tuple) else output
            captured['h']=h[0,-1].float().clone()
        forward(enc,capture)
        return captured['h']

    def intervention(mode,q,donor=None,kind='natural'):
        def hook(_module,_inputs,output):
            h=output[0] if isinstance(output,tuple) else output
            last=h[0,-1].float()
            if mode=='ablate':
                delta=((last-center)@q)@q.T
                if kind=='norm_matched':
                    budget=args.norm_fraction*(last-center).norm().clamp_min(1e-8)
                    delta=delta/delta.norm().clamp_min(1e-8)*budget
                change=-args.alpha*delta
            elif mode=='swap':
                delta=((donor-last)@q)@q.T
                if kind=='norm_matched':
                    budget=args.norm_fraction*(donor-last).norm().clamp_min(1e-8)
                    delta=delta/delta.norm().clamp_min(1e-8)*budget
                change=args.alpha*delta
            else:raise ValueError(mode)
            altered=h.clone()
            altered[0,-1]=(last+change).to(h.dtype)
            return replace_hidden(output,altered)
        return hook

    ablate_rows,swap_rows=[],[]
    for task in ('nli','language_id'):
        for i,row in enumerate(samples):
            gold=int(row['label']) if task=='nli' else LANG_LABELS[row['language']]
            enc=encode(row,task)
            original=forward(enc)
            baseline=score(original,gold)
            ablate_rows.append(dict(task=task,example_id=row['example_id'],
                                    language=row['language'],subspace='none',
                                    variant='none',rank=0,**baseline))
            donor_row=donor_for(samples,row,task)
            donor=get_last_hidden(encode(donor_row,task)) if donor_row else None
            donor_label=(int(donor_row['label']) if task=='nli'
                         else LANG_LABELS[donor_row['language']]) if donor_row else None
            for name,q in basis.items():
                for variant in ('natural','norm_matched'):
                    altered=forward(enc,intervention('ablate',q,kind=variant))
                    ablate_rows.append(dict(task=task,example_id=row['example_id'],
                                            language=row['language'],subspace=name,
                                            variant=variant,rank=q.shape[1],**score(altered,gold)))
                    if donor is not None:
                        swapped=forward(enc,intervention('swap',q,donor=donor,kind=variant))
                        orig_margin=float(torch.log(original[donor_label].clamp_min(1e-12))-torch.log(original[gold].clamp_min(1e-12)))
                        swap_margin=float(torch.log(swapped[donor_label].clamp_min(1e-12))-torch.log(swapped[gold].clamp_min(1e-12)))
                        swap_rows.append(dict(task=task,example_id=row['example_id'],
                            language=row['language'],subspace=name,variant=variant,rank=q.shape[1],
                            donor_id=donor_row['example_id'],donor_label=donor_label,gold_label=gold,
                            donor_margin_change=swap_margin-orig_margin,
                            donor_prob_change=float(swapped[donor_label]-original[donor_label]),
                            target_gold_prob_change=float(swapped[gold]-original[gold])))
            print(f'[causal] {task} {i+1}/{len(samples)}',flush=True)

    write_csv(out/'ablation_example_level.csv',ablate_rows)
    write_csv(out/'swap_example_level.csv',swap_rows)
    baselines={(r['task'],r['example_id']):r for r in ablate_rows if r['subspace']=='none'}
    aggregated=[]
    for name in basis:
        for task in ('nli','language_id'):
            for variant in ('natural','norm_matched'):
                selected=[r for r in ablate_rows if r['subspace']==name and r['task']==task and r['variant']==variant]
                if not selected:continue
                base=[baselines[(r['task'],r['example_id'])] for r in selected]
                swaps=[r for r in swap_rows if r['subspace']==name and r['task']==task and r['variant']==variant]
                aggregated.append(dict(
                    subspace=name,rank=basis[name].shape[1],task=task,variant=variant,n=len(selected),
                    baseline_accuracy=float(np.mean([r['accuracy'] for r in base])),
                    altered_accuracy=float(np.mean([r['accuracy'] for r in selected])),
                    accuracy_change=float(np.mean([r['accuracy']-b['accuracy'] for r,b in zip(selected,base)])),
                    baseline_nll=float(np.mean([r['gold_nll'] for r in base])),
                    altered_nll=float(np.mean([r['gold_nll'] for r in selected])),
                    nll_increase=float(np.mean([r['gold_nll']-b['gold_nll'] for r,b in zip(selected,base)])),
                    gold_probability_drop=float(np.mean([b['gold_p']-r['gold_p'] for r,b in zip(selected,base)])),
                    swap_donor_margin_increase=float(np.mean([r['donor_margin_change'] for r in swaps])) if swaps else None,
                    swap_pairs=len(swaps)))
    write_csv(out/'causal_summary.csv',aggregated)
    write_json(out/'causal_protocol.json',dict(
        checkpoint=args.checkpoint,subspace_file=args.subspace_file,layer=layer,
        token='last prompt token only',
        feature_mismatch='Bases fitted using mean-pooled states; intervention edits last token only.',
        n_examples_requested=args.n_examples,n_examples_actual=len(samples),
        norm_fraction=args.norm_fraction,alpha=args.alpha,
        no_conclusive_semantic_identification=True))
    bacc=float(np.mean([r['accuracy'] for r in ablate_rows if r['subspace']=='none' and r['task']=='nli']))
    print(f'[causal] NLI baseline accuracy={bacc:.3f}; chance=0.333; interpret carefully.',flush=True)
    print(f'CAUSAL COMPLETE: {out}',flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--subspace_file',required=True)
    ap.add_argument('--bases_file',required=True)
    ap.add_argument('--probe_data',required=True)
    ap.add_argument('--out_dir',required=True)
    ap.add_argument('--n_examples',type=int,default=36)
    ap.add_argument('--max_length',type=int,default=512)
    ap.add_argument('--alpha',type=float,default=1.0)
    ap.add_argument('--norm_fraction',type=float,default=0.03)
    ap.add_argument('--seed',type=int,default=2026)
    ap.add_argument('--only',type=str,default='')
    a=ap.parse_args()
    if a.n_examples<9 or a.norm_fraction<=0 or a.alpha<=0:
        ap.error('n_examples >= 9, alpha and norm_fraction positive')
    run(a)

if __name__=='__main__':main()
