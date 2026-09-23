"""Equal-count geometry screening on train annotations; no neural computation."""
from pathlib import Path
from datetime import datetime
import argparse,hashlib,json,math,time
import xml.etree.ElementTree as ET
import numpy as np

ROOT=Path(__file__).resolve().parent

def grids(h,w,ratio):
    rows=[]
    for size in (64,96,128,160,192,224,256):
        rw=round(size*math.sqrt(ratio));rh=round(size/math.sqrt(ratio))
        rows.extend([x,y,x+rw-1,y+rh-1] for y in range(0,h-rh+1,32) for x in range(0,w-rw+1,32))
    return np.asarray(rows,dtype=float).reshape(-1,4)

def matched_regions(h,w,image_id,seed,protect_small=False):
    pools=[grids(h,w,r) for r in (1.,.5,2.)];n=len(pools[0]);assert n
    original=pools[0]
    protected=np.empty((0,4))
    if protect_small:
        small=(original[:,2]-original[:,0]+1)<=96
        protected=original[small];pools[0]=original[~small]
        pools[1:]=[p[(p[:,2:]-p[:,:2]+1).prod(1)>=127**2] for p in pools[1:]]
    budget=n-len(protected)
    # Keep half square; split the remainder equally between two aspect ratios.
    quotas=[budget//2,(budget-budget//2)//2,budget-budget//2-(budget-budget//2)//2]
    quotas=[min(q,len(p)) for q,p in zip(quotas,pools)]
    for i,p in enumerate(pools):
        extra=min(budget-sum(quotas),len(p)-quotas[i]);quotas[i]+=extra
    rng=np.random.default_rng(int.from_bytes(hashlib.sha256(f'{seed}:{image_id}'.encode()).digest()[:8],'little'))
    mixed=np.concatenate([protected]+[p[rng.choice(len(p),q,replace=False)] for p,q in zip(pools,quotas) if q])
    assert len(mixed)==n and len(np.unique(mixed,axis=0))==n
    assert (mixed[:,:2]>=0).all() and (mixed[:,2]<w).all() and (mixed[:,3]<h).all()
    return original,mixed

def best_iou(regions,box):
    inter=np.maximum(0,np.minimum(regions[:,2:],box[2:])-np.maximum(regions[:,:2],box[:2])+1).prod(1)
    return float(np.max(inter/((regions[:,2:]-regions[:,:2]+1).prod(1)+(box[2:]-box[:2]+1).prod()-inter)))

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--protect-small",action="store_true");args=parser.parse_args()
    start=time.perf_counter();run=ROOT/'runs'/f'step23_equal_budget_{datetime.now():%Y%m%d_%H%M%S}';run.mkdir()
    def dump(n,v):(run/n).write_text(json.dumps(v,indent=2),encoding='utf-8')
    voc=ROOT/'datasets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012'
    ids=(voc/'ImageSets/Main/train.txt').read_text().split();val=set((voc/'ImageSets/Main/val.txt').read_text().split());assert not set(ids)&val
    seeds=[42,43,44];dump('plan.json',dict(split='all VOC2012 train',seeds=seeds,protect_small=args.protect_small,selection='all non-difficult objects of every class',
        budget='same count as existing square grid per image',mix='Protect all64/96 squares if requested; remaining budget:50% squares; 25% ratio0.5; remainder ratio2; uniform no-replacement per pool',
        image_scale=480,step=32,annotation_use='evaluation only; sampler receives h,w,id,seed and never object boxes',
        criterion='weak macro mean best-IoU +0.02 and no other class mean loss >0.02; no AP claims'))
    rows=[];counts=[]
    for ident in ids:
        tree=ET.parse(voc/'Annotations'/f'{ident}.xml');ow=int(tree.findtext('size/width'));oh=int(tree.findtext('size/height'))
        factor=480/max(ow,oh);w=int(ow*factor+.5);h=int(oh*factor+.5)
        pairs=[matched_regions(h,w,ident,s,args.protect_small) for s in seeds];square=pairs[0][0];counts.append(len(square))
        for obj in tree.findall('object'):
            if int(obj.findtext('difficult','0')):continue
            b=obj.find('bndbox');raw=np.array([float(b.findtext(k)) for k in ('xmin','ymin','xmax','ymax')]);box=raw.copy()
            box[:2]=(raw[:2]-1)*[w/ow,h/oh];box[2:]=raw[2:]*[w/ow,h/oh]-1
            fraction=float(np.prod(box[2:]-box[:2]+1)/(w*h))
            rows.append(dict(id=ident,cls=obj.findtext('name'),size='small' if fraction<.05 else 'medium' if fraction<.2 else 'large',
                square=best_iou(square,box),mixed=[best_iou(p[1],box) for p in pairs]))
    summary={}
    for cls in sorted(set(r['cls'] for r in rows)):
        items=[r for r in rows if r['cls']==cls];a=np.array([r['square'] for r in items]);b=np.array([r['mixed'] for r in items])
        summary[cls]=dict(objects=len(items),square_mean=float(a.mean()),mixed_mean=float(b.mean()),delta=float(b.mean()-a.mean()),
             seed_deltas=(b.mean(0)-a.mean()).tolist(),square_coverage50=float((a>=.5).mean()),mixed_coverage50=float((b>=.5).mean()),sizes={})
        for size in ('small','medium','large'):
            subset=[r for r in items if r['size']==size]
            if subset:summary[cls]['sizes'][size]=dict(n=len(subset),delta=float(np.mean([np.mean(r['mixed'])-r['square'] for r in subset])))
    weak=('bottle','pottedplant','sofa');gain=float(np.mean([summary[c]['delta'] for c in weak]));worst=min(v['delta'] for c,v in summary.items() if c not in weak)
    dump('result.json',dict(status='completed',images=len(ids),objects=len(rows),neural_updates=0,model_forward_calls=0,
        mean_regions=float(np.mean(counts)),summary=summary,weak_macro_delta=gain,worst_other_delta=worst,
        proceed_to_feature_probe=bool(gain>=.02 and worst>=-.02),elapsed_seconds=time.perf_counter()-start,
        limitation='Geometry of oracle best region, not learned scores. Three sampling seeds are sensitivity checks, not independent datasets.'))
    dump('objects.json',rows);(run/'step23_equal_budget_regions.py').write_text(Path(__file__).read_text(encoding='utf-8'),encoding='utf-8')
    print('RUN',run);print(json.dumps(dict(weak={c:summary[c] for c in weak},weak_macro_delta=gain,worst_other_delta=worst),indent=2))

if __name__=='__main__':main()
