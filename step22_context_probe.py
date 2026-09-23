"""Train-only paired context intervention on fixed shared-SPP regions; no training."""
from pathlib import Path
from datetime import datetime
import json,time
import xml.etree.ElementTree as ET
import numpy as np
import torch
from data import CLASSES
from data.patch_inputs import VOCPatchDataset,dense_regions
from models.fisher_classifier import FisherClassifier
from step13_evaluate import sha

ROOT=Path(__file__).resolve().parent
WEAK=('bottle','pottedplant','sofa')

def main():
    start=time.perf_counter();torch.set_num_threads(2);torch.manual_seed(42)
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    run=ROOT/'runs'/f'step22_context_probe_{datetime.now():%Y%m%d_%H%M%S}';run.mkdir()
    def dump(n,v):(run/n).write_text(json.dumps(v,indent=2),encoding='utf-8')
    ds=VOCPatchDataset(ROOT/'datasets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012','train')
    rng=np.random.default_rng(42);selected=[]
    for cls in WEAK:
        candidates=np.flatnonzero(ds.raw_labels[:,CLASSES.index(cls)].numpy()==1)
        selected.extend((cls,int(i)) for i in rng.permutation(candidates)[:8])
    dump('plan.json',dict(seed=42,samples_per_class=8,selection='train positive label, seeded sample; largest non-difficult box; highest IoU existing dense square',
         samples=[dict(cls=c,image_id=ds.ids[i]) for c,i in selected],
         interventions=['original','outside ROI -> normalized zero','outside ROI+32px halo -> normalized zero','independent ROI crop resized224'],
         caveat='Boxes are diagnostic annotations only. Mask boundaries and crop resize are confounds. No AP or model promotion from descriptor distance.'))
    ckpath=ROOT/'runs/step12_scale_20260921_130318/best.pt';digest=sha(ckpath)
    model,_,_=FisherClassifier.from_gmm(ROOT/'artifacts/alexnet_baseline_v1',ROOT/'runs/step08_gmm_20260920_014414_978178','cuda')
    ck=torch.load(ckpath,map_location='cuda',weights_only=False);model.load_state_dict(ck['model']);del ck
    model.eval();model.requires_grad_(False);encoder=model.encoder.patch_encoder
    rows=[]
    with torch.inference_mode():
        for cls,i in selected:
            sample=ds[i];image=sample['image'].cuda();h,w=image.shape[-2:];sx,sy=sample['scale_xy']
            tree=ET.parse(ds.root/'Annotations'/f'{ds.ids[i]}.xml');boxes=[]
            for obj in tree.findall('object'):
                if obj.findtext('name')!=cls or int(obj.findtext('difficult','0')):continue
                b=obj.find('bndbox');v=[float(b.findtext(k))-1 for k in ('xmin','ymin','xmax','ymax')]
                boxes.append([v[0]*sx,v[1]*sy,(v[2]+1)*sx-1,(v[3]+1)*sy-1])
            box=max(boxes,key=lambda b:(b[2]-b[0]+1)*(b[3]-b[1]+1))
            regions=dense_regions((h,w),device='cuda');xy=regions[:,1:];b=torch.tensor(box,device='cuda')
            inter=(torch.minimum(xy[:,2:],b[2:])-torch.maximum(xy[:,:2],b[:2])+1).clamp_min(0).prod(1)
            area=(xy[:,2:]-xy[:,:2]+1).prod(1);ba=(b[2:]-b[:2]+1).prod();iou=inter/(area+ba-inter)
            k=int(iou.argmax());roi=regions[k:k+1];x1,y1,x2,y2=map(int,roi[0,1:].tolist())
            base,info=encoder(image[None],roi,True)
            variants={}
            for halo in (0,32):
                xa,ya,xb,yb=max(0,x1-halo),max(0,y1-halo),min(w,x2+halo+1),min(h,y2+halo+1)
                changed=torch.zeros_like(image);changed[:,ya:yb,xa:xb]=image[:,ya:yb,xa:xb]
                assert torch.equal(changed[:,y1:y2+1,x1:x2+1],image[:,y1:y2+1,x1:x2+1])
                variants[f'mask_halo{halo}']=encoder(changed[None],roi)
            crop=torch.nn.functional.interpolate(image[None,:,y1:y2+1,x1:x2+1],size=(224,224),mode='bilinear',align_corners=False,antialias=True)
            variants['crop224']=encoder(crop,torch.tensor([[0.,0.,0.,223.,223.]],device='cuda'))
            row=dict(cls=cls,image_id=ds.ids[i],roi=roi[0].cpu().tolist(),object_box=box,roi_iou=float(iou[k]),
                     object_area_fraction=float(ba/(h*w)),roi_object_fraction=float(inter[k]/area[k]),
                     projected_roi=info['projected_regions'][0].cpu().tolist())
            for name,desc in variants.items():
                row[name]=dict(cosine=float(torch.nn.functional.cosine_similarity(base,desc)),relative_l2=float((desc-base).norm()/base.norm().clamp_min(1e-12)))
            if not rows:
                repeated=encoder(image.clone()[None],roi);torch.testing.assert_close(repeated,base,rtol=0,atol=0)
            rows.append(row)
    summary={}
    for cls in (*WEAK,'all'):
        items=[r for r in rows if cls=='all' or r['cls']==cls]
        summary[cls]={n:dict(median_cosine=float(np.median([r[n]['cosine'] for r in items])),median_relative_l2=float(np.median([r[n]['relative_l2'] for r in items]))) for n in variants}
        summary[cls]['median_roi_iou']=float(np.median([r['roi_iou'] for r in items]))
    assert sha(ckpath)==digest
    dump('result.json',dict(status='completed',neural_updates=0,checkpoint_sha256=digest,rows=rows,summary=summary,
         roi_pixels_preserved=True,no_op_repeat_exact=True,checkpoint_unchanged=True,elapsed_seconds=time.perf_counter()-start))
    (run/'step22_context_probe.py').write_text(Path(__file__).read_text(encoding='utf-8'),encoding='utf-8')
    print('RUN',run);print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
