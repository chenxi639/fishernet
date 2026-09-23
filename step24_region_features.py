"""Frozen equal-budget mixed-region probe, paired with cached square descriptors."""
import json,time
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from data.patch_inputs import VOCPatchDataset
from data import CLASSES
from models.fisher_classifier import FisherClassifier,normalize_fisher
from metrics import classification_map
from step23_equal_budget_regions import matched_regions
from step20_train_only_selection import predict

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'runs/step19_model_audit_20260922_230655'

def main():
    start=time.perf_counter();torch.set_num_threads(2);torch.manual_seed(42)
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    run=ROOT/'runs'/f'step24_region_features_{datetime.now():%Y%m%d_%H%M%S}';run.mkdir()
    def dump(n,v):(run/n).write_text(json.dumps(v,indent=2),encoding='utf-8')
    dump('plan.json',dict(source=str(SOURCE),checkpoint='epoch6',seed=42,train_images=256,val_images=256,
        sampler='protect all64/96 squares; equal count; no boxes',probe='ridge alpha1; no tuning',
        caveat='Previously used diagnostic subsets; encoder trained on full train; exploratory only.'))
    model,_,_=FisherClassifier.from_gmm(ROOT/'artifacts/alexnet_baseline_v1',ROOT/'runs/step08_gmm_20260920_014414_978178','cuda')
    ck=torch.load(ROOT/'runs/step12_scale_20260921_130318/best.pt',map_location='cuda',weights_only=False)
    model.load_state_dict(ck['model']);del ck;model.eval();model.requires_grad_(False)
    features={};labels={};ids={};drifts=[];head_scores={}
    for split in ('train','val'):
        cache=np.load(SOURCE/f'{split}_features.npz');ids[split]=cache['image_ids'].tolist();labels[split]=cache['labels']
        ds=VOCPatchDataset(ROOT/'datasets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012',split);xs=[];zs=[]
        with torch.inference_mode():
            for j,ident in enumerate(ids[split]):
                sample=ds[ds.ids.index(ident)];np.testing.assert_array_equal(sample['raw_label'].numpy(),labels[split][j]);im=sample['image'].cuda()
                square,mixed=matched_regions(*im.shape[-2:],ident,42,True)
                roi=torch.cat([torch.zeros(len(mixed),1),torch.from_numpy(mixed).float()],1).cuda()
                desc=model.encoder.patch_encoder(im[None],roi,region_chunk_size=64)
                f=normalize_fisher(model.encoder.fisher(desc[None]));xs.append(f.cpu().numpy()[0]);zs.append(model.head(f).cpu().numpy()[0])
                if j==0:
                    base=normalize_fisher(model.encoder([im])['fisher']).cpu().numpy()[0]
                    np.testing.assert_allclose(base,cache['fisher'][0],rtol=2e-4,atol=2e-5)
            features[split]=np.array(xs);head_scores[split]=np.array(zs)
        drifts.append(dict(split=split,mean_cosine=float((features[split]*cache['fisher']).sum(1).mean())))
        np.savez(run/f'{split}_features.npz',fisher=features[split],labels=labels[split],image_ids=np.array(ids[split]))
        print('EXTRACTED',split,flush=True)
    assert not set(ids['train'])&set(ids['val']);results={}
    with threadpool_limits(limits=2):
        for mode in ('square','mixed'):
            x=(np.load(SOURCE/'train_features.npz')['fisher'] if mode=='square' else features['train']).astype(float)
            v=(np.load(SOURCE/'val_features.npz')['fisher'] if mode=='square' else features['val']).astype(float)
            s=predict(x@x.T,v@x.T,labels['train'],np.arange(len(x)),np.arange(len(v)),1.)
            m=classification_map(labels['val'],s);results[mode]=dict(map=float(m['map']),ap=dict(zip(CLASSES,m['ap'].tolist())))
            np.savez(run/f'{mode}_predictions.npz',scores=s,labels=labels['val'],image_ids=np.array(ids['val']))
        m=classification_map(labels['val'],head_scores['val']);results['old_head_on_mixed']=dict(map=float(m['map']),note='Distribution shift diagnostic only; head not adapted.')
    weak=('bottle','pottedplant','sofa')
    dump('result.json',dict(status='completed',results=results,drifts=drifts,neural_updates=0,probe_class_fits=40,
        delta_map_pp=100*(results['mixed']['map']-results['square']['map']),
        weak_mean_delta_pp=float(np.mean([100*(results['mixed']['ap'][c]-results['square']['ap'][c]) for c in weak])),
        elapsed_seconds=time.perf_counter()-start))
    (run/'step24_region_features.py').write_text(Path(__file__).read_text(encoding='utf-8'),encoding='utf-8')
    print('RUN',run);print(json.dumps(results),flush=True)

if __name__=='__main__':main()
