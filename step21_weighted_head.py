"""Matched frozen-feature head experiment on complete VOC train/val caches."""
from pathlib import Path
from datetime import datetime
import json,time
import numpy as np
import torch
from data import CLASSES
from metrics import classification_map
from step13_evaluate import sha

ROOT=Path(__file__).resolve().parent
CACHE=ROOT/'runs/step13_eval_20260921_170016'
WEAK=('bottle','pottedplant','sofa')

def main():
    start=time.perf_counter();run=ROOT/'runs'/f'step21_weighted_head_{datetime.now():%Y%m%d_%H%M%S}';run.mkdir()
    def dump(n,v):(run/n).write_text(json.dumps(v,indent=2),encoding='utf-8')
    plan=dict(epochs=20,batch_size=128,lr=.0003,optimizer='Adam',seed=42,weak_positive_weight=1.5,
              weak_classes=WEAK,features='frozen epoch6 mean480_576',encoder_updates=0,
              normalization='weighted sum BCE divided by sum of valid entry weights',
              evaluation='final epoch only; no validation-driven early stopping',
              gates='candidate vs matched control: mAP +0.10pp, weak mean +0.50pp, no other class below -1pp; candidate vs original head mAP >= -0.05pp',
              limitation='Full val reused historically; exploratory, not independent confirmation.')
    dump('plan.json',plan);print('RUN',run,flush=True)
    torch.set_num_threads(2);torch.manual_seed(42);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    # Small linear heads run on CPU for deterministic BLAS without CUDA workspace setup.
    cache_config=json.loads((CACHE/'config.json').read_text())
    ckpath=Path(cache_config['checkpoint']);assert sha(ckpath)==cache_config['checkpoint_sha256']
    ck=torch.load(ckpath,map_location='cpu',mmap=True,weights_only=False)
    arrays={};labels={};ids={};checks={}
    for split in ('train','val'):
        ids[split]=json.loads((CACHE/f'{split}_ids.json').read_text());assert ids[split]==ck[f'{split}_ids']
        parts=[]
        for scale in (480,576):
            p=CACHE/f'{split}_{scale}.npy';marker=json.loads((CACHE/f'{split}_{scale}_complete.json').read_text())
            assert sha(p)==marker['sha256'];checks[p.name]=marker['sha256'];parts.append(np.load(p,mmap_mode='r'))
        arrays[split]=torch.from_numpy((parts[0]+parts[1])*.5)
        labels[split]=torch.from_numpy(np.load(CACHE/f'{split}_labels.npy'))
    assert not set(ids['train'])&set(ids['val'])
    assert len(ids['train'])==5717 and len(ids['val'])==5823
    x=arrays['train'];y=labels['train'];target=(y==1).float();valid=y!=0
    weight=torch.ones_like(target)
    for name in WEAK:
        c=CLASSES.index(name);weight[y[:,c]==1,c]=1.5
    histories={};predictions={};metrics={}
    initial_w=ck['model']['head.weight'].clone();initial_b=ck['model']['head.bias'].clone();del ck
    with torch.no_grad():
        initial=torch.nn.functional.linear(arrays['val'],initial_w,initial_b).numpy()
    reference=np.load(CACHE/'mean480_576_neural_predictions.npz')
    np.testing.assert_array_equal(reference['image_ids'],np.array(ids['val']))
    np.testing.assert_array_equal(reference['labels'],labels['val'].numpy())
    np.testing.assert_allclose(initial,reference['scores'],atol=2e-5,rtol=2e-4)
    predictions['original']=initial
    for name,w in [('control',torch.ones_like(weight)),('weighted',weight)]:
        head=torch.nn.Linear(16384,20);head.weight.data.copy_(initial_w);head.bias.data.copy_(initial_b)
        opt=torch.optim.Adam(head.parameters(),lr=plan['lr']);rng=np.random.default_rng(42);history=[]
        for epoch in range(20):
            order=rng.permutation(len(x));total=0.;denom=0.
            for j in range(0,len(x),128):
                ii=order[j:j+128];entry_weight=w[ii]*valid[ii]
                element=torch.nn.functional.binary_cross_entropy_with_logits(head(x[ii]),target[ii],reduction='none')
                loss=(element*entry_weight).sum()/entry_weight.sum()
                assert torch.isfinite(loss);opt.zero_grad();loss.backward();opt.step()
                total+=float((element.detach()*entry_weight).sum());denom+=float(entry_weight.sum())
            history.append(total/denom)
        with torch.no_grad():predictions[name]=head(arrays['val']).numpy()
        torch.save(dict(weight=head.weight.detach(),bias=head.bias.detach(),plan=plan),run/f'{name}_head.pt')
        histories[name]=history;print('HEAD_DONE',name,flush=True)
    for name,scores in predictions.items():
        m=classification_map(labels['val'].numpy(),scores)
        metrics[name]=dict(map=float(m['map']),ap=dict(zip(CLASSES,m['ap'].tolist())))
        np.savez(run/f'{name}_predictions.npz',scores=scores,labels=labels['val'].numpy(),image_ids=np.array(ids['val']))
    delta=np.array([100*(metrics['weighted']['ap'][c]-metrics['control']['ap'][c]) for c in CLASSES])
    wi=[CLASSES.index(c) for c in WEAK];other=[i for i,c in enumerate(CLASSES) if c not in WEAK]
    rng=np.random.default_rng(42);boot=[]
    for _ in range(500):
        ii=rng.integers(0,len(labels['val']),len(labels['val']));y=labels['val'].numpy()[ii]
        boot.append(100*(classification_map(y,predictions['weighted'][ii])['map']-classification_map(y,predictions['control'][ii])['map']))
    gates=dict(matched_map_gain=bool(delta.mean()>=.1),weak_mean_gain=bool(delta[wi].mean()>=.5),
               no_large_other_drop=bool(delta[other].min()>=-1),
               original_head_preserved=bool(100*(metrics['weighted']['map']-metrics['original']['map'])>=-.05))
    dump('result.json',dict(status='completed',metrics=metrics,matched_delta_map_pp=float(delta.mean()),
         weak_mean_delta_pp=float(delta[wi].mean()),per_class_delta_pp=dict(zip(CLASSES,delta.tolist())),
         bootstrap95_pp=np.percentile(boot,[2.5,97.5]).tolist(),gates=gates,recommend_joint_trial=all(gates.values()),
         histories=histories,head_updates_per_arm=900,encoder_updates=0,cache_checks=checks,
         elapsed_seconds=time.perf_counter()-start))
    (run/'step21_weighted_head.py').write_text(Path(__file__).read_text(encoding='utf-8'),encoding='utf-8')
    print(json.dumps(dict(metrics=metrics,gates=gates)),flush=True)

if __name__=='__main__':main()
