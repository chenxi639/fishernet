"""Frozen-model representation audit; diagnostic subsets, not benchmark scores."""
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from data.patch_inputs import VOCPatchDataset
from data import CLASSES
from models.fisher_classifier import FisherClassifier, normalize_fisher
from metrics import classification_map
from step04_train import subset_indices

ROOT=Path(__file__).resolve().parent

def main():
    torch.set_num_threads(2);torch.manual_seed(42)
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    run=ROOT/'runs'/f'step19_model_audit_{datetime.now():%Y%m%d_%H%M%S}'
    run.mkdir();start=time.perf_counter()
    def dump(name,obj):(run/name).write_text(json.dumps(obj,indent=2),encoding='utf-8')
    checkpoint=ROOT/'runs/step12_scale_20260921_130318/best.pt'
    model,_,_=FisherClassifier.from_gmm(ROOT/'artifacts/alexnet_baseline_v1',ROOT/'runs/step08_gmm_20260920_014414_978178','cuda')
    ck=torch.load(checkpoint,map_location='cuda',weights_only=False)
    model.load_state_dict(ck['model']);del ck
    model.eval();model.requires_grad_(False)
    modes=('mean','max','first','second','fisher','intra','temperature2')
    all_x={};all_y={};health={};head_scores={};split_ids={}
    for split,seed in [('train',42),('val',43)]:
        ds=VOCPatchDataset(ROOT/'datasets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012',split)
        indices=subset_indices(ds,256,seed);split_ids[split]=[ds.ids[i] for i in indices]
        xs={m:[] for m in modes};ys=[];logits=[];records=[];occupancy=torch.zeros(32,device='cuda')
        with torch.inference_mode():
            for j,i in enumerate(indices):
                sample=ds[i];out=model.encoder([sample['image'].cuda()]);x=out['flat_descriptors'];f=out['fisher']
                values=dict(mean=torch.nn.functional.normalize(x.mean(0)[None],dim=-1),
                            max=torch.nn.functional.normalize(x.amax(0)[None],dim=-1),
                            first=normalize_fisher(f[:,:8192]),second=normalize_fisher(f[:,8192:]),
                            fisher=normalize_fisher(f),intra=normalize_fisher(f,mode='intra'))
                values['temperature2']=normalize_fisher(model.encoder.fisher(out['descriptors'],out['mask'],assignment_temperature=2.))
                for mode in modes:xs[mode].append(values[mode].cpu().numpy()[0])
                logits.append(model.head(values['fisher']).cpu().numpy()[0]);ys.append(sample['raw_label'].numpy())
                z=model.encoder.fisher.w[None]*(x[:,None]+model.encoder.fisher.b[None])
                gamma=torch.softmax(-.5*z.square().sum(-1),-1);occupancy+=gamma.sum(0)
                entropy=-(gamma*gamma.clamp_min(1e-30).log()).sum(-1)
                soft_gamma=torch.softmax(-.25*z.square().sum(-1),-1)
                roi=out['details'][0]['projected_regions'];wh=roi[:,3:5]-roi[:,1:3]+1
                block=values['fisher'].reshape(2,32,256).square().sum((0,2))
                records.append(dict(patches=len(x),unique_regions=out['details'][0]['unique_projected_regions'],
                                    pooled_roi_below6_fraction=float((wh.min(-1).values<6).float().mean()),
                                    assignment_entropy=float(entropy.mean()),assignment_max=float(gamma.max(-1).values.mean()),
                                    temperature2_assignment_max=float(soft_gamma.max(-1).values.mean()),
                                    top_component_energy=float(block.max()/block.sum()),
                                    descriptor_within_image_std=float(x.std(0).mean())))
                if (j+1)%64==0:print(split,j+1,flush=True)
        all_x[split]={m:np.array(xs[m]) for m in modes};all_y[split]=np.array(ys);head_scores[split]=np.array(logits)
        p=(occupancy/occupancy.sum()).cpu().numpy()
        health[split]=dict(mean={k:float(np.mean([r[k] for r in records])) for k in records[0]},
                           component_mass=p.tolist(),effective_components=float(np.exp(-(p*np.log(p+1e-30)).sum())),
                           records=records)
        np.savez(run/f'{split}_features.npz',**all_x[split],labels=all_y[split],image_ids=np.array(split_ids[split]))
    assert not set(split_ids['train'])&set(split_ids['val'])
    # Train-only centered ridge regression, common unit-norm features and fixed alpha.
    # Fit each class on valid entries; no hyperparameter search on val.
    results={}
    with threadpool_limits(limits=2):
        for mode in modes:
            x=all_x['train'][mode].astype(np.float64);v=all_x['val'][mode].astype(np.float64)
            train_scores=np.zeros((len(x),20));val_scores=np.zeros((len(v),20))
            for c in range(20):
                keep=all_y['train'][:,c]!=0;t=(all_y['train'][keep,c]==1).astype(float)
                center=x[keep].mean(0);a=x[keep]-center;bias=t.mean()
                dual=np.linalg.solve(a@a.T+np.eye(len(a)),t-bias)
                w=a.T@dual
                train_scores[:,c]=(x-center)@w+bias;val_scores[:,c]=(v-center)@w+bias
            ta=classification_map(all_y['train'],train_scores);va=classification_map(all_y['val'],val_scores)
            results[mode]=dict(dim=x.shape[1],train_map=float(ta['map']),val_map=float(va['map']),val_ap=dict(zip(CLASSES,va['ap'].tolist())))
            np.savez(run/f'{mode}_probe_predictions.npz',scores=val_scores,labels=all_y['val'],image_ids=np.array(split_ids['val']))
    for split in ('train','val'):
        a=classification_map(all_y[split],head_scores[split]);results[f'original_head_{split}']=dict(map=float(a['map']),ap=dict(zip(CLASSES,a['ap'].tolist())))
    dump('result.json',dict(status='completed',checkpoint=str(checkpoint),neural_updates=0,linear_probe_fits=20*len(modes),
         protocol='256 class-covered train / 256 class-covered val; scale480; frozen epoch6; ridge alpha1; no tuning',
         results=results,health=health,split_ids=split_ids,elapsed_seconds=time.perf_counter()-start,
         limitations=['Subset scores are not full-val results.','Feature dimensions differ; common ridge alpha does not equalize effective model capacity.',
                      'Encoder already trained on full train: probe train score is not an independent generalization estimate.',
                      'Intra normalization is evaluated with a refitted probe, not the old classifier head.']))
    (run/'step19_model_audit.py').write_text(Path(__file__).read_text(encoding='utf-8'),encoding='utf-8')
    print('RUN',run,flush=True);print(json.dumps(results),flush=True)

if __name__=='__main__':main()
