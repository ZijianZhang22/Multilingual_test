#!/usr/bin/env python3
"""7B last-token model-level ablation and donor swapping; frozen adapted checkpoint.

WARNING: LAST bases were fit on unprompted XNLI including some test inputs.
Behavioral prompts contain task instructions, so these are exploratory results.
"""
from __future__ import annotations
import argparse,csv,json,random
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

LANGS=('en','zh','fr','de','es')
TASKS=('nli','language_id')
DEFAULT_ROOT='mechanism_runs/semantic_causal_validation/qwen25_7b_semantic_seed0'
DEFAULT_CKPT='replication_runs/qwen25_7b_a100_fresh_fullzh/seed0/lr_4e-05/adapted'

def read_jsonl(p):
    with open(p,encoding='utf-8') as f:
        return [json.loads(s) for s in f if s.strip()]

def save_csv(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)

def save_json(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')

def choose_balanced(rows,n,seed):
    if n<=0 or n%(len(LANGS)*3):
        raise ValueError('n_examples must be a multiple of 15, e.g. 30,60,150,300')
    groups=defaultdict(list)
    for r in rows:
        if r.get('split')=='probe_test' and r.get('language') in LANGS and int(r.get('label',-1)) in (0,1,2):
            groups[(r['language'],int(r['label']))].append(r)
    rng=random.Random(seed);selected=[]
    for lang in LANGS:
        for label in range(3):
            opts=groups[(lang,label)];per=n//15
            if len(opts)<per:raise RuntimeError(f'Insufficient samples: {lang} {label}')
            selected.extend(rng.sample(opts,per))
    rng.shuffle(selected)
    return selected

def pick_donor(rows,r,task):
    if task=='nli':
        options=[x for x in rows if x['language']==r['language'] and int(x['label'])!=int(r['label'])]
    else:
        options=[x for x in rows if x['language']!=r['language'] and int(x['label'])==int(r['label'])]
    if not options:raise RuntimeError('No matched donor')
    return options[0]

def prompt(r,task):
    if task=='nli':
        return ('Read the premise and hypothesis. Choose one letter: '
                'A = entailment, B = neutral, C = contradiction.\n'
                f'Premise: {r["premise"]}\nHypothesis: {r["hypothesis"]}\nAnswer:')
    if task=='language_id':
        return ('Identify the language used in the premise. Choose one letter: '
                'A = English, B = Chinese, C = French, D = German, E = Spanish.\n'
                f'Premise: {r["premise"]}\nAnswer:')
    raise ValueError(task)

def token_ids(tok,task):
    letters='ABC' if task=='nli' else 'ABCDE'
    ids=[tok.encode(' '+c,add_special_tokens=False) for c in letters]
    if any(len(x)!=1 for x in ids) or len({x[0] for x in ids})!=len(letters):
        raise RuntimeError(f'Options are not distinct single-token continuations: {ids}')
    return [x[0] for x in ids]

def basis(q):
    q=torch.as_tensor(q,dtype=torch.float32,device='cpu')
    return torch.linalg.qr(q,mode='reduced')[0]

def load_bases(core_file,part_file,names,seed,device):
    core=torch.load(core_file,map_location='cpu',weights_only=False)
    part=torch.load(part_file,map_location='cpu',weights_only=False)
    if core.get('pool')!='last':raise ValueError('Must use last-token fitted core, not mean')
    if int(core['layer'])!=int(part['layer']):raise ValueError('Layer mismatch')
    available={k:basis(v) for k,v in core['subspaces'].items() if not k.startswith('random_')}
    available.update({k.replace('drift_isr_',''):basis(v) for k,v in part['derived_subspaces'].items()})
    dim=int(core['hidden_dim'])
    for rank in (16,32,64):
        rng=torch.Generator().manual_seed(seed+rank)
        available[f'random{rank}']=basis(torch.randn(dim,rank,generator=rng))
    chosen=[s.strip() for s in names.split(',') if s.strip()]
    unknown=set(chosen)-set(available)
    if unknown:raise ValueError(f'Unknown subspaces {sorted(unknown)}')
    return core,{k:available[k].to(device) for k in chosen}

def project(x,q):
    return (x@q)@q.T

def match_scales(parts):
    ranks=defaultdict(list)
    for name,(rank,v) in parts.items():ranks[rank].append(float(v.norm()))
    target={r:min(v) for r,v in ranks.items()}
    return {name:min(1.0,target[r]/max(float(v.norm()),1e-15)) for name,(r,v) in parts.items()}

def gold(r,task):
    return int(r['label']) if task=='nli' else LANGS.index(r['language'])

def metric(logits,y):
    lp=torch.log_softmax(logits.float(),dim=-1)
    return dict(accuracy=int(logits.argmax().item()==y),
                gold_nll=float(-lp[y]),gold_p=float(lp[y].exp()))

def aggregate(ablation,swaps):
    baselines={(r['task'],r['example_id']):r for r in ablation if r['mode']=='baseline'}
    groups=defaultdict(list);sgroups=defaultdict(list)
    for r in ablation:
        if r['mode']!='baseline':groups[(r['task'],r['subspace'],r['mode'],r['beta'])].append(r)
    for r in swaps:sgroups[(r['task'],r['subspace'],r['mode'],r['beta'])].append(r)
    result=[]
    for (task,name,mode,beta),rows in sorted(groups.items()):
        old=[baselines[(task,r['example_id'])] for r in rows]
        s=sgroups[(task,name,mode,beta)]
        result.append(dict(task=task,subspace=name,rank=rows[0]['rank'],mode=mode,beta=beta,
            n=len(rows),baseline_acc=float(np.mean([r['accuracy'] for r in old])),
            ablated_acc=float(np.mean([r['accuracy'] for r in rows])),
            acc_change=float(np.mean([x['accuracy']-y['accuracy'] for x,y in zip(rows,old)])),
            nll_increase=float(np.mean([x['gold_nll']-y['gold_nll'] for x,y in zip(rows,old)])),
            mean_actual_energy=float(np.mean([r['delta_l2'] for r in rows])),
            language_id_fr_acc=(float(np.mean([r['accuracy'] for r in rows if r['language']=='fr'])) if task=='language_id' else ''),
            donor_margin_increase=(float(np.mean([r['donor_margin_change'] for r in s])) if s else ''),
            donor_prob_increase=(float(np.mean([r['donor_probability_change'] for r in s])) if s else '')))
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',default=DEFAULT_CKPT)
    ap.add_argument('--run_root',default=DEFAULT_ROOT)
    ap.add_argument('--probe_data',default='')
    ap.add_argument('--out_dir',default='')
    ap.add_argument('--n_examples',type=int,default=30)
    ap.add_argument('--only',default='top16,bottom16,random16,top32,bottom32,random32')
    ap.add_argument('--betas',nargs='+',type=float,default=[1.0])
    ap.add_argument('--seed',type=int,default=2026)
    ap.add_argument('--max_length',type=int,default=768)
    ap.add_argument('--save_every',type=int,default=5)
    a=ap.parse_args()
    if not a.betas or any(not 0<b<=1 for b in a.betas):ap.error('betas must be in (0,1]')
    if a.save_every<1:ap.error('save_every must be >=1')
    root=Path(a.run_root)
    cf=root/'last_pool_subspaces/core_subspaces.pt'
    pf=root/'last_pool_drift_isr_partition/drift_isr_partition.pt'
    data=Path(a.probe_data) if a.probe_data else root/'last_pool_subspaces/data/xnli_probe.jsonl'
    out=Path(a.out_dir) if a.out_dir else root/'model_level_lasttoken'/f'n{a.n_examples}_seed{a.seed}'
    out.mkdir(parents=True,exist_ok=True)
    for p in (cf,pf,data,Path(a.checkpoint)/'config.json'):
        if not p.is_file():raise FileNotFoundError(p)
    samples=choose_balanced(read_jsonl(data),a.n_examples,a.seed)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for 7B model-level forward passes')
    device=torch.device('cuda')
    core,qs=load_bases(cf,pf,a.only,a.seed,device)
    from transformers import AutoTokenizer,AutoModelForCausalLM
    tok=AutoTokenizer.from_pretrained(a.checkpoint,use_fast=True)
    ids={t:torch.tensor(token_ids(tok,t),device=device) for t in TASKS}
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model=AutoModelForCausalLM.from_pretrained(a.checkpoint,torch_dtype=dtype,
        low_cpu_mem_usage=True,attn_implementation='sdpa').to(device).eval()
    model.config.use_cache=False
    for p in model.parameters():p.requires_grad_(False)
    layer=int(core['layer'])
    block=model.model.layers[layer-1]
    center=core['center'].to(device=device,dtype=torch.float32)
    if center.numel()!=model.config.hidden_size:raise ValueError('Checkpoint dimensions do not match')
    # FP32 readout from only the 3/5 candidate LM-head rows avoids BF16 rounding.
    weights={t:model.lm_head.weight.detach().index_select(0,ids[t]).float() for t in TASKS}
    biases={t:(model.lm_head.bias.detach().index_select(0,ids[t]).float()
             if model.lm_head.bias is not None else None) for t in TASKS}

    def forward(row,task,alter=None,capture=False):
        state={}
        def hook(_module,_input,output):
            h=output[0] if isinstance(output,tuple) else output
            if capture:state['hidden']=h[0,-1].detach().float().clone()
            if alter is None:return None
            z=h.clone()
            z[0,-1]=(h[0,-1].float()+alter).to(h.dtype)
            return (z,*output[1:]) if isinstance(output,tuple) else z
        handle=block.register_forward_hook(hook) if (capture or alter is not None) else None
        try:
            inp=tok(prompt(row,task),return_tensors='pt',truncation=False)
            if inp['input_ids'].shape[1]>a.max_length:
                raise RuntimeError(f'Prompt too long: {row["example_id"]}')
            inp={k:v.to(device) for k,v in inp.items()}
            with torch.inference_mode():
                last=model.model(**inp,use_cache=False,return_dict=True).last_hidden_state[0,-1].float()
                logits=last@weights[task].T
                if biases[task] is not None:logits=logits+biases[task]
                return logits.detach().cpu(),state.get('hidden')
        finally:
            if handle is not None:handle.remove()

    baseline={};ablation=[];swaps=[]
    for task in TASKS:
        for i,row in enumerate(samples):
            logits,h=forward(row,task,capture=True)
            baseline[(task,row['example_id'])]=(logits,h)
            ablation.append(dict(task=task,example_id=row['example_id'],
                language=row['language'],label=int(row['label']),subspace='none',rank=0,
                mode='baseline',beta=0.0,scale=0.0,delta_l2=0.0,**metric(logits,gold(row,task))))
            if i==0:
                zero,_=forward(row,task,alter=torch.zeros_like(h))
                error=float((zero-logits).abs().max())
                print(f'[sanity] {task}: zero-hook max logit error {error:.6g}',flush=True)
                if error>.05:raise RuntimeError('Zero-hook must not change logits')
        acc=np.mean([r['accuracy'] for r in ablation if r['task']==task])
        print(f'[baseline] {task}: accuracy {acc:.3f} n={len(samples)}',flush=True)

    save_json(out/'causal_protocol.json',dict(checkpoint=a.checkpoint,layer=layer,pool='last',
        core_file=str(cf),partition_file=str(pf),dataset=str(data),n_examples=len(samples),
        languages=list(LANGS),betas=a.betas,bases=list(qs),seed=a.seed,
        intervention_token='last prompted token at transformer block output',
        natural='Unscaled projected component',matched='per example and rank, shrink all projections to min natural L2',
        logits='FP32 readout from restricted LM-head rows, actual language model forward pass',
        limitation_1='Transductive fitted Drift includes probe_test features',
        limitation_2='Subspaces fitted on unprompted premise+hypothesis, tested on task prompts',
        limitation_3='Donor swapping not a controlled semantic counterfactual'))

    def persist(status):
        save_csv(out/'ablation_example_level.csv',ablation)
        save_csv(out/'swap_example_level.csv',swaps)
        save_csv(out/'causal_summary.csv',aggregate(ablation,swaps))
        save_json(out/'run_status.json',status)

    for task in TASKS:
        for i,row in enumerate(samples):
            rid=row['example_id'];y=gold(row,task)
            old_logits,h=baseline[(task,rid)]
            donor=pick_donor(samples,row,task)
            dh=baseline[(task,donor['example_id'])][1]
            dy=gold(donor,task)
            a_parts={k:(q.shape[1],project(h-center,q)) for k,q in qs.items()}
            s_parts={k:(q.shape[1],project(dh-h,q)) for k,q in qs.items()}
            a_scales=match_scales(a_parts);s_scales=match_scales(s_parts)
            old_logp=torch.log_softmax(old_logits.float(),dim=-1)
            old_margin=float(old_logp[dy]-old_logp[y]);old_dp=float(old_logp[dy].exp())
            for name,q in qs.items():
                for mode in ('natural','matched'):
                    for beta in a.betas:
                        scale=1.0 if mode=='natural' else a_scales[name]
                        delta=-beta*scale*a_parts[name][1]
                        logits,_=forward(row,task,alter=delta)
                        ablation.append(dict(task=task,example_id=rid,language=row['language'],
                            label=int(row['label']),subspace=name,rank=q.shape[1],mode=mode,beta=beta,
                            scale=scale,delta_l2=float(delta.norm()),**metric(logits,y)))
                        scale_s=1.0 if mode=='natural' else s_scales[name]
                        delta_s=beta*scale_s*s_parts[name][1]
                        swap_logits,_=forward(row,task,alter=delta_s)
                        sp=torch.log_softmax(swap_logits.float(),dim=-1)
                        swaps.append(dict(task=task,example_id=rid,language=row['language'],
                            label=int(row['label']),subspace=name,rank=q.shape[1],mode=mode,beta=beta,
                            scale=scale_s,delta_l2=float(delta_s.norm()),donor_id=donor['example_id'],
                            donor_language=donor['language'],donor_label=int(donor['label']),
                            donor_margin_change=float((sp[dy]-sp[y])-old_margin),
                            donor_probability_change=float(sp[dy].exp())-old_dp))
            if (i+1)%a.save_every==0 or i+1==len(samples):
                persist(dict(status='running',task=task,processed=i+1,total=len(samples)))
            print(f'[model-level] {task} {i+1}/{len(samples)}',flush=True)
    persist(dict(status='complete',n_examples=len(samples),tasks=list(TASKS)))
    print('[complete] '+str(out),flush=True)
    for r in aggregate(ablation,swaps):
        if r['subspace'] in ('top32','bottom32','random32') and r['mode']=='matched' and r['beta']==1.0:
            print(f'{r["task"]:12s} {r["subspace"]:10s} acc_delta={r["acc_change"]:+.3f} '
                  f'nll_delta={r["nll_increase"]:+.5f} swap_margin={r["donor_margin_increase"]:+.5f}',flush=True)

if __name__=='__main__':
    main()
