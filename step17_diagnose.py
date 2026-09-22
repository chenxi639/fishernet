"""Read-only model diagnosis from saved predictions and VOC annotations; no fit."""
import json,time
from pathlib import Path
from datetime import datetime
import xml.etree.ElementTree as ET
import numpy as np
from data.voc_classification import VOCClassification,CLASSES
from metrics import classification_map
ROOT=Path(__file__).resolve().parent
WEAK=('bottle','pottedplant','sofa')
def dump(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')
def main():
 start=time.perf_counter();run=ROOT/'runs'/f'step17_diagnosis_{datetime.now():%Y%m%d_%H%M%S}';run.mkdir();print('RUN',run,flush=True)
 root=ROOT/'datasets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012'
 datasets={s:VOCClassification(root,s) for s in ('train','val')}
 old=np.load(ROOT/'runs/step14_688_20260921_172409/triple_predictions.npz')
 new=np.load(ROOT/'runs/step11_lrbranch_20260922_094801/triple_evaluation/predictions.npz')
 assert np.array_equal(old['image_ids'],new['image_ids']) and new['image_ids'].tolist()==datasets['val'].ids
 assert np.array_equal(old['labels'],new['labels']) and np.array_equal(new['labels'],datasets['val'].raw_labels.numpy())
 sampled=set(json.loads((ROOT/'runs/step07_sample_20260920_012954_361930/image_ids.json').read_text()))
 assert sampled<=set(datasets['train'].ids) and not set(datasets['train'].ids)&set(datasets['val'].ids)
 annotation={};size_stats={};coverage={};manifest=[];diagnosis={}
 for split,ds in datasets.items():
  records=[]
  for ident in ds.ids:
   x=ET.parse(root/'Annotations'/f'{ident}.xml').getroot();w=int(x.findtext('size/width'));h=int(x.findtext('size/height'))
   objects=[]
   for obj in x.findall('object'):
    bb=obj.find('bndbox');box=[int(float(bb.findtext(k))) for k in ('xmin','ymin','xmax','ymax')]
    objects.append(dict(name=obj.findtext('name'),difficult=int(obj.findtext('difficult','0')),truncated=int(obj.findtext('truncated','0')),box=box))
   record=dict(id=ident,width=w,height=h,objects=objects);records.append(record);annotation[ident]=record
  size_stats[split]={}
  labels=ds.raw_labels.numpy()
  coverage[split]={name:int((labels[:,k]==1).sum()) for k,name in enumerate(CLASSES)}
  for name in WEAK:
   k=CLASSES.index(name);values=[]
   for i in np.flatnonzero(labels[:,k]==1):
    rec=records[i];objects=[o for o in rec['objects'] if o['name']==name and not o['difficult']]
    assert objects,(split,rec['id'],name)
    largest=max(objects,key=lambda o:(o['box'][2]-o['box'][0]+1)*(o['box'][3]-o['box'][1]+1))
    x1,y1,x2,y2=largest['box'];area=(x2-x1+1)*(y2-y1+1)/(rec['width']*rec['height'])
    values.append(dict(index=int(i),id=rec['id'],area=area,truncated=bool(largest['truncated']),
     short_at480=min(x2-x1+1,y2-y1+1)*480/max(rec['width'],rec['height'])))
   size_stats[split][name]=values
  print('ANNOTATIONS',split,len(records),flush=True)
 for name in WEAK:
  k=CLASSES.index(name);y=new['labels'][:,k];s=new['scores'][:,k];s0=old['scores'][:,k]
  positive=y==1;negative=y==-1;prediction=s>=0
  tp=int((positive&prediction).sum());fp=int((negative&prediction).sum());fn=int((positive&~prediction).sum())
  groups={}
  for label,lo,hi in [('small',0,.05),('medium',.05,.2),('large',.2,2)]:
   groups[label]={}
   for split in ('train','val'):
    subset=[v for v in size_stats[split][name] if lo<=v['area']<hi]
    entry=dict(count=len(subset),fraction=len(subset)/len(size_stats[split][name]))
    if split=='val':
     ii=[v['index'] for v in subset];entry.update(recall_epoch6=float((s0[ii]>=0).mean()),recall_epoch7=float((s[ii]>=0).mean()))
    groups[label][split]=entry
  geometry={}
  for split in ('train','val'):
   values=size_stats[split][name];geometry[split]=dict(median_area=float(np.median([v['area'] for v in values])),
    short_below64_at480=sum(v['short_at480']<64 for v in values),positive_count=len(values),
    largest_object_truncated=sum(v['truncated'] for v in values),gmm_sample_positive=sum(v['id'] in sampled for v in values) if split=='train' else None)
  context=[]
  for j,other in enumerate(CLASSES):
   if j==k:continue
   present=new['labels'][:,j]==1;known_absent=new['labels'][:,j]==-1
   n=int((negative&present).sum());absent_n=int((negative&known_absent).sum())
   if n>=30 and absent_n:
    hits=int((negative&present&prediction).sum());absent_hits=int((negative&known_absent&prediction).sum())
    context.append(dict(other=other,negative_with_other=n,fp_with_other=hits,fpr=hits/n,
     negative_without_other=absent_n,fp_without_other=absent_hits,fpr_without=absent_hits/absent_n))
  context.sort(key=lambda v:v['fpr'],reverse=True)
  newly_missed=np.flatnonzero(positive&(s0>=0)&(s<0));recovered=np.flatnonzero(positive&(s0<0)&(s>=0))
  diagnosis[name]=dict(tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp),recall=tp/(tp+fn),groups=groups,geometry=geometry,
   contexts=context[:5],new_misses_epoch7=[str(new['image_ids'][i]) for i in newly_missed],recovered_epoch7=[str(new['image_ids'][i]) for i in recovered])
  for typ,indices in [('FN',np.flatnonzero(positive&~prediction)),('FP',np.flatnonzero(negative&prediction))]:
   ranked=indices[np.argsort(s[indices],kind='stable')]
   if typ=='FP':ranked=ranked[::-1]
   for i in ranked[:4]:
    ident=str(new['image_ids'][i]);rec=annotation[ident]
    manifest.append(dict(target=name,kind=typ,id=ident,score_epoch7=float(s[i]),score_epoch6=float(s0[i]),
     image=str(root/'JPEGImages'/f'{ident}.jpg'),width=rec['width'],height=rec['height'],objects=rec['objects']))
 dump(run/'selected_errors.json',manifest)
 print('DIAGNOSIS',json.dumps(diagnosis),flush=True)
 # Paired image bootstrap quantifies conditional validation sampling sensitivity.
 rng=np.random.default_rng(42);boot=[];y=new['labels'];s=new['scores'];s0=old['scores']
 for iteration in range(500):
  ii=rng.integers(0,len(y),len(y));a=classification_map(y[ii],s0[ii]);b=classification_map(y[ii],s[ii])
  boot.append((b['ap']-a['ap'])*100)
  if (iteration+1)%100==0:print('BOOTSTRAP',iteration+1,flush=True)
 boot=np.array(boot);np.save(run/'paired_bootstrap_ap_deltas.npy',boot)
 uncertainty=dict(replicates=500,seed=42,unit='image paired across checkpoints',
  overall_delta_pp=float((classification_map(y,s)['map']-classification_map(y,s0)['map'])*100),
  overall_percentile95_pp=np.percentile(boot.mean(axis=1),[2.5,97.5]).tolist(),
  weak_percentile95_pp={name:np.percentile(boot[:,CLASSES.index(name)],[2.5,97.5]).tolist() for name in WEAK},
  caveat='Conditional on these checkpoints and reused validation set; not training-seed uncertainty or independent test evidence.')
 result=dict(status='completed',neural_updates=0,model_forward_calls=0,svm_fits=0,train_images=5717,val_images=5823,
  coverage=coverage,weak_classes=diagnosis,uncertainty=uncertainty,gmm_sample_images=len(sampled),
  assumptions=['Positive image assigned largest non-difficult target box; small area<5%, medium5-20%,large>=20%.',
   'Recall uses logit>=0; context groups overlap and indicate association, not causality.',
   'Contact sheets contain four lowest-score false negatives and four highest-score false positives per class; not representative samples.'],
  elapsed_seconds=time.perf_counter()-start)
 dump(run/'result.json',result);dump(run/'annotation_size_groups.json',size_stats)
 print('COMPLETED',run,json.dumps(uncertainty),flush=True)
if __name__=='__main__':main()
