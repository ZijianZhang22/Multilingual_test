#!/usr/bin/env python3
"""XNLI held-out NLI/language probes and aligned cross-lingual retrieval."""
import argparse,re
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from shared import make_bases,read_jsonl,write_csv,write_json,normalized

def probe_fit_predict(x_train,x_test,y_train,y_test,seed=0):
    if len(np.unique(y_train))<2:
        return {'accuracy':float('nan'),'balanced_accuracy':float('nan'),'macro_f1':float('nan')}
    m=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=1200,random_state=seed))
    m.fit(x_train,y_train);yp=m.predict(x_test)
    return dict(accuracy=float(accuracy_score(y_test,yp)),
                balanced_accuracy=float(balanced_accuracy_score(y_test,yp)),
                macro_f1=float(f1_score(y_test,yp,average='macro',zero_division=0)))

def probe_all(bases,feat,out_dir,seed):
    split=np.asarray(feat['splits'])
    if not np.any(split=='probe_train') or not np.any(split=='probe_test'):
        raise ValueError('Missing train/test probe split')
    x=feat['features'][str(feat['layer'])].float()
    labels=np.asarray(feat['labels'].tolist());langs=np.asarray(feat['languages'])
    tr=split=='probe_train';te=split=='probe_test'
    rng=np.random.default_rng(seed);shuffle=labels[tr].copy();rng.shuffle(shuffle)
    rows=[]
    for name,q in bases.items():
        z=(x@q).numpy()
        for task,ya,yb in [('nli',labels[tr],labels[te]),
                           ('nli_permuted_train_control',shuffle,labels[te]),
                           ('language_id',langs[tr],langs[te])]:
            rows.append(dict(subspace=name,rank=q.shape[1],task=task,
                             train_langs='all',test_langs='all',
                             n_train=int(tr.sum()),n_test=int(te.sum()),
                             **probe_fit_predict(z[tr],z[te],ya,yb,seed)))
        for lang in sorted(set(langs)):
            tm=te & (langs==lang)
            if tm.sum()<5:continue
            rows.append(dict(subspace=name,rank=q.shape[1],task='nli_by_language',
                             train_langs='all',test_langs=lang,n_train=int(tr.sum()),
                             n_test=int(tm.sum()),**probe_fit_predict(z[tr],z[tm],labels[tr],labels[tm],seed)))
            lm=tr & (langs!=lang)
            if lm.sum()>10:
                rows.append(dict(subspace=name,rank=q.shape[1],task='nli_probe_leave_language_out',
                                 train_langs='except_'+lang,test_langs=lang,n_train=int(lm.sum()),n_test=int(tm.sum()),
                                 **probe_fit_predict(z[lm],z[tm],labels[lm],labels[tm],seed)))
        print('[probe]',name,flush=True)
    write_csv(out_dir/'probe_results.csv',rows)
    write_json(out_dir/'probe_metadata.json',dict(train='XNLI validation',test='XNLI test',
        caveat='LOO only excludes language from the probe classifier, not from the pre-fitted subspace.'))
    return rows

def terms(row):
    return set(re.findall(r"[\w']+",(row['premise']+' '+row['hypothesis']).lower()))

def retrieval_all(bases,feat,data_rows,out_dir,max_pairs=256,seed=0):
    if len(data_rows)!=len(feat['languages']):raise ValueError('Aligned rows/features mismatch')
    x=feat['features'][str(feat['layer'])].float()
    pairs={}
    for i,row in enumerate(data_rows):
        if row['split']!='aligned':raise ValueError('Expected aligned rows')
        pairs.setdefault(str(row['pair_id']),{})[row['language']]=i
    languages=sorted(set(feat['languages']));rng=np.random.default_rng(seed);results=[]
    for la in languages:
        for lb in languages:
            if la==lb:continue
            ids=[p for p,m in pairs.items() if la in m and lb in m]
            if len(ids)<8:continue
            rng.shuffle(ids);ids=ids[:max_pairs];n=len(ids)
            ai=[pairs[p][la] for p in ids];bi=[pairs[p][lb] for p in ids]
            pivot=[data_rows[pairs[p].get('en',pairs[p][la])] for p in ids]
            lex=[terms(v) for v in pivot];label=[int(data_rows[i]['label']) for i in ai]
            hard=[]
            for i in range(n):
                cand=[j for j in range(n) if j!=i and label[i]==label[j]] or [j for j in range(n) if j!=i]
                hard.append(max(cand,key=lambda j: len(lex[i]&lex[j])/max(len(lex[i]|lex[j]),1)-.001*abs(len(lex[i])-len(lex[j]))))
            for name,q in bases.items():
                scores=(normalized(x[ai]@q)@normalized(x[bi]@q).T).numpy()
                ranks=np.argsort(-scores,axis=1)
                position=np.asarray([int(np.where(ranks[i]==i)[0][0])+1 for i in range(n)])
                margins=np.asarray([scores[i,i]-scores[i,hard[i]] for i in range(n)])
                results.append(dict(subspace=name,rank=q.shape[1],query_language=la,target_language=lb,
                    pairs=n,recall1=float(np.mean(position==1)),recall5=float(np.mean(position<=5)),
                    mrr=float(np.mean(1/position)),hard_negative_accuracy=float(np.mean(margins>0)),
                    hard_negative_similarity_margin=float(np.mean(margins)),
                    hard_negative_basis='English-pivot Jaccard and same XNLI label'))
            print(f'[retrieval] {la}->{lb}: {n}',flush=True)
    write_csv(out_dir/'retrieval_results.csv',results)
    return results

def main():
    ap=argparse.ArgumentParser()
    for key in ('subspace_file','probe_features','aligned_features','aligned_data','out_dir'):
        ap.add_argument('--'+key,required=True)
    ap.add_argument('--max_pairs',type=int,default=256)
    ap.add_argument('--seed',type=int,default=2026)
    a=ap.parse_args()
    pr=torch.load(a.probe_features,map_location='cpu',weights_only=False)
    test=torch.load(a.aligned_features,map_location='cpu',weights_only=False)
    core=torch.load(a.subspace_file,map_location='cpu',weights_only=False)
    layer=str(core['layer']);pr['layer']=layer;test['layer']=layer
    x=pr['features'][layer][np.asarray(pr['splits'])=='probe_train']
    bases,_=make_bases(a.subspace_file,train_x=x,seed=a.seed)
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    torch.save(dict(layer=int(layer),bases=bases),out/'validated_bases.pt')
    write_json(out/'bases_manifest.json',{k:int(v.shape[1]) for k,v in bases.items()})
    probe_all(bases,pr,out,a.seed)
    retrieval_all(bases,test,read_jsonl(a.aligned_data),out,a.max_pairs,a.seed)
    print('PROBES/RETRIEVAL COMPLETE:',out,flush=True)

if __name__=='__main__':main()
