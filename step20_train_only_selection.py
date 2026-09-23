"""Train-only OOF selection on frozen cached representations; no CNN updates."""
import json
import hashlib
from pathlib import Path
from datetime import datetime
import numpy as np
from threadpoolctl import threadpool_limits
from metrics import classification_map
from data import CLASSES

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'runs/step19_model_audit_20260922_230655'

def predict(kernel, cross, labels, train, test, alpha):
    scores=np.zeros((len(test),20))
    for c in range(20):
        idx=train[labels[train,c]!=0]
        y=(labels[idx,c]==1).astype(float)
        if y.min()==y.max():raise ValueError('each fitting fold needs positive and negative labels')
        k=kernel[np.ix_(idx,idx)]; means=k.mean(0); overall=k.mean()
        centered=k-means[None,:]-means[:,None]+overall
        dual=np.linalg.solve(centered+alpha*np.eye(len(idx)),y-y.mean())
        v=cross[np.ix_(test,idx)]
        scores[:,c]=(v-v.mean(1,keepdims=True)-means[None,:]+overall)@dual+y.mean()
    return scores

def main():
    run=ROOT/'runs'/f'step20_train_selection_{datetime.now():%Y%m%d_%H%M%S}';run.mkdir()
    def dump(name,obj):(run/name).write_text(json.dumps(obj,indent=2),encoding='utf-8')
    cache=np.load(SOURCE/'train_features.npz');y=cache['labels'];n=len(y)
    rng=np.random.default_rng(42);folds=np.empty(n,dtype=int);folds[rng.permutation(n)]=np.arange(n)%4
    modes=('fisher','first','temperature2');alphas=(.1,1.,10.)
    dump('plan.json',dict(source=str(SOURCE),modes=modes,alphas=alphas,folds=4,seed=42,
         fold_assignment=folds.tolist(),train_ids=cache['image_ids'].tolist(),
         selection='highest pooled OOF macro AP; tie uses listed order; val not loaded until selection saved',
         caveat='Encoder saw full train. OOF is independent only of probe fitting, not representation learning.'))
    candidates=[];kernels={};features={}
    with threadpool_limits(limits=2):
        for mode in modes:
            x=cache[mode].astype(np.float64);features[mode]=x;k=x@x.T;kernels[mode]=k
            for alpha in alphas:
                scores=np.zeros_like(y,dtype=float)
                for fold in range(4):
                    train=np.flatnonzero(folds!=fold);test=np.flatnonzero(folds==fold)
                    scores[test]=predict(k,k,y,train,test,alpha)
                metric=classification_map(y,scores)
                row=dict(mode=mode,alpha=alpha,oof_map=float(metric['map']),ap=dict(zip(CLASSES,metric['ap'].tolist())))
                candidates.append(row);np.savez(run/f'oof_{mode}_{alpha}.npz',scores=scores,labels=y)
                print(mode,alpha,row['oof_map'],flush=True)
        best=max(candidates,key=lambda d:d['oof_map']);dump('selection.json',dict(selected=best,candidates=candidates))
        # Only now read already-used diagnostic val: no iterative retuning after this check.
        val=np.load(SOURCE/'val_features.npz');assert not set(cache['image_ids'])&set(val['image_ids'])
        checks={}
        for name,mode,alpha in [('selected',best['mode'],best['alpha']),('baseline','fisher',1.)]:
            x=features[mode];v=val[mode].astype(float)
            scores=predict(kernels[mode],v@x.T,y,np.arange(n),np.arange(len(v)),alpha)
            metric=classification_map(val['labels'],scores)
            checks[name]=dict(mode=mode,alpha=alpha,map=float(metric['map']),ap=dict(zip(CLASSES,metric['ap'].tolist())))
            np.savez(run/f'{name}_val_predictions.npz',scores=scores,labels=val['labels'],image_ids=val['image_ids'])
            if name=='selected':
                weights=[];biases=[]
                for c in range(20):
                    keep=y[:,c]!=0;t=(y[keep,c]==1).astype(float);center=x[keep].mean(0);a=x[keep]-center
                    w=a.T@np.linalg.solve(a@a.T+alpha*np.eye(len(a)),t-t.mean())
                    weights.append(w);biases.append(t.mean()-center@w)
                w=np.array(weights);b=np.array(biases)
                np.testing.assert_allclose(v@w.T+b,scores,atol=1e-10,rtol=1e-9)
                np.savez(run/'candidate_probe.npz',weight=w,bias=b,class_order=np.array(CLASSES))
    delta=(checks['selected']['map']-checks['baseline']['map'])*100
    dump('result.json',dict(status='completed',neural_updates=0,oof_class_fits=720,final_class_fits=40,export_refits=20,
         selected=best,diagnostic_val=checks,delta_val_pp=delta,benchmark_promoted=False,
         source_train_sha256=hashlib.sha256((SOURCE/'train_features.npz').read_bytes()).hexdigest(),
         limitation='256-image diagnostic val was used previously; not full val or independent confirmation. Candidate probe is ridge scores, not calibrated probabilities.'))
    (run/'step20_train_only_selection.py').write_text(Path(__file__).read_text(encoding='utf-8'),encoding='utf-8')
    print('RUN',run,flush=True);print('SELECTED',best['mode'],best['alpha'],'VAL_DELTA_PP',delta,flush=True)

if __name__=='__main__':main()
