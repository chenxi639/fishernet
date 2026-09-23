"""Locked mixed sampler, frozen epoch6 head, full val480 comparison."""
from pathlib import Path
from datetime import datetime
import json,time
import numpy as np
import torch
from models.fisher_classifier import FisherClassifier,normalize_fisher
from data.patch_inputs import VOCPatchDataset
from data import CLASSES
from metrics import classification_map
from step23_equal_budget_regions import matched_regions

ROOT=Path(__file__).resolve().parent
def main():
    start=time.perf_counter();torch.set_num_threads(2);torch.manual_seed(42)
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    run=ROOT/'runs'/f'step24_full_val_{datetime.now():%Y%m%d_%H%M%S}';run.mkdir()
    def dump(n,v):(run/n).write_text(json.dumps(v,indent=2),encoding='utf-8')
    dump('plan.json',dict(checkpoint='epoch6',scale=480,sampler='protect-small equal count',seed=42,
         classifier='original checkpoint head unchanged',training=False,
         limitation='Full val already used repeatedly; post-subset exploratory confirmation, not untouched test.'))
    model,_,_=FisherClassifier.from_gmm(ROOT/'artifacts/alexnet_baseline_v1',ROOT/'runs/step08_gmm_20260920_014414_978178','cuda')
    ck=torch.load(ROOT/'runs/step12_scale_20260921_130318/best.pt',map_location='cuda',weights_only=False)
    model.load_state_dict(ck['model']);del ck;model.eval();model.requires_grad_(False)
    ds=VOCPatchDataset(ROOT/'datasets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012','val');scores=[]
    reference=np.load(ROOT/'runs/step12_scale_20260921_130318/best_predictions.npz');assert reference['image_ids'].tolist()==ds.ids
    with torch.inference_mode():
        for i in range(len(ds)):
            s=ds[i];image=s['image'].cuda();square,mixed=matched_regions(*image.shape[-2:],s['image_id'],42,True)
            roi=torch.cat([torch.zeros(len(mixed),1),torch.from_numpy(mixed).float()],1).cuda()
            desc=model.encoder.patch_encoder(image[None],roi,region_chunk_size=64)
            f=normalize_fisher(model.encoder.fisher(desc[None]));z=model.head(f);assert torch.isfinite(z).all();scores.append(z.cpu().numpy()[0])
            if (i+1)%512==0:print('PROGRESS',i+1,len(ds),flush=True)
    scores=np.array(scores);labels=ds.raw_labels.numpy();np.testing.assert_array_equal(labels,reference['labels'])
    a=classification_map(labels,scores);b=classification_map(labels,reference['scores']);delta=100*(a['ap']-b['ap'])
    np.savez(run/'predictions.npz',scores=scores,labels=labels,image_ids=np.array(ds.ids))
    rng=np.random.default_rng(42);boot=[]
    for _ in range(500):
        ii=rng.integers(0,len(ds),len(ds));boot.append(100*(classification_map(labels[ii],scores[ii])['map']-classification_map(labels[ii],reference['scores'][ii])['map']))
    result=dict(status='completed',neural_updates=0,baseline_map=float(b['map']),mixed_map=float(a['map']),delta_map_pp=float(delta.mean()),
        ap=dict(zip(CLASSES,a['ap'].tolist())),ap_delta_pp=dict(zip(CLASSES,delta.tolist())),bootstrap95_pp=np.percentile(boot,[2.5,97.5]).tolist(),
        weak_mean_delta_pp=float(np.mean([delta[CLASSES.index(c)] for c in ('bottle','pottedplant','sofa')])),elapsed_seconds=time.perf_counter()-start)
    dump('result.json',result);(run/'step24_full_val.py').write_text(Path(__file__).read_text(encoding='utf-8'),encoding='utf-8')
    print('RUN',run);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
