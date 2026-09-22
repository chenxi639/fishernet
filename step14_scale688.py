"""Frozen epoch-6 neural head: add scale 688 using verified existing caches."""
import json,time
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from data.patch_inputs import VOCPatchDataset
from data.voc_classification import CLASSES
from models.fisher_classifier import FisherClassifier,normalize_fisher
from metrics import classification_map
from step13_evaluate import sha,dump
ROOT=Path(__file__).resolve().parent

def main():
 old=ROOT/'runs/step13_eval_20260921_170016';prior=json.loads((old/'config.json').read_text())
 checkpoint=Path(prior['checkpoint']);digest=sha(checkpoint);assert digest==prior['checkpoint_sha256']
 run=ROOT/'runs'/f'step14_688_{datetime.now():%Y%m%d_%H%M%S}';run.mkdir()
 start=time.perf_counter();print('RUN',run,flush=True)
 cfg=dict(checkpoint=str(checkpoint),checkpoint_sha256=digest,source_cache=str(old),scales=[480,576,688],
  pooling=prior['pooling'],classifier='unchanged checkpoint neural head',neural_updates=0,svm_fits=0,
  split='VOC2012 val5823',batch_size=2,seed=42,augmentation='none; eval mode',normalization_epsilon=1e-6)
 dump(run/'config.json',cfg);dump(run/'result.json',dict(status='running'))
 try:
  features={}
  for scale in (480,576):
   path=old/f'val_{scale}.npy';meta=json.loads((old/f'val_{scale}_complete.json').read_text());assert sha(path)==meta['sha256']
   features[scale]=np.load(path,mmap_mode='r');assert features[scale].shape==(5823,16384)
  ids=json.loads((old/'val_ids.json').read_text());labels=np.load(old/'val_labels.npy')
  torch.manual_seed(42);torch.set_num_threads(2);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
  device='cuda' if torch.cuda.is_available() else 'cpu'
  model,_,_=FisherClassifier.from_gmm(ROOT/'artifacts/alexnet_baseline_v1',ROOT/'runs/step08_gmm_20260920_014414_978178',device)
  ck=torch.load(checkpoint,map_location=device,weights_only=False);model.load_state_dict(ck['model']);assert ck['val_ids']==ids;del ck
  model.eval();model.requires_grad_(False)
  w=model.head.weight.detach().cpu().numpy().copy();b=model.head.bias.detach().cpu().numpy().copy()
  root=ROOT/'datasets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012'
  ds={s:VOCPatchDataset(root,'val',s) for s in (480,576,688)}
  assert all(d.ids==ids and np.array_equal(d.raw_labels.numpy(),labels) for d in ds.values())
  dump(run/'val_ids.json',ids);np.save(run/'val_labels.npy',labels)
  # Matched timing sample, equal images and repeated measurements for each scale.
  indices=np.sort(np.random.default_rng(42).choice(len(ids),64,replace=False)).tolist()
  timing={};dump(run/'benchmark_ids.json',[ids[i] for i in indices])
  def sync():
   if device=='cuda':torch.cuda.synchronize()
  with torch.inference_mode():
   for s in (480,576,688):
    model([ds[s][i]['image'].to(device) for i in indices[:2]]);sync()
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    times=[]
    for repeat in range(2):
     sync();t=time.perf_counter()
     for j in range(0,len(indices),2):model([ds[s][i]['image'].to(device) for i in indices[j:j+2]])
     sync();times.append(time.perf_counter()-t)
    timing[str(s)]=dict(seconds_for_64=times,median_seconds_per_image=float(np.median(times)/64),
     peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2 if device=='cuda' else None)
    print('BENCHMARK',s,timing[str(s)],flush=True)
   x=np.lib.format.open_memmap(run/'val_688.npy',mode='w+',dtype=np.float32,shape=(len(ids),16384));t=time.perf_counter()
   for j in range(0,len(ids),2):
    images=[ds[688][i]['image'].to(device) for i in range(j,min(j+2,len(ids)))]
    f=normalize_fisher(model.encoder(images)['fisher']);assert torch.isfinite(f).all()
    if j==0:torch.testing.assert_close(model(images),model.head(f),rtol=0,atol=0)
    x[j:j+len(images)]=f.cpu().numpy()
    if (j+2)%512==0 or j+2>=len(ids):
     progress=dict(processed=min(j+2,len(ids)),total=len(ids),phase='extract688',elapsed_seconds=time.perf_counter()-start)
     dump(run/'progress.json',progress);print('PROGRESS',progress,flush=True)
   sync();seconds=time.perf_counter()-t;x.flush();del x
  dump(run/'val_688_complete.json',dict(sha256=sha(run/'val_688.npy'),shape=[len(ids),16384],seconds=seconds))
  features[688]=np.load(run/'val_688.npy',mmap_mode='r')
  for j in range(0,len(ids),128):np.testing.assert_allclose(np.linalg.norm(features[688][j:j+128],axis=1),1,atol=2e-6)
  variants={};score_cache={}
  for name,scales in [('dual',[480,576]),('single688',[688]),('triple',[480,576,688])]:
   f=sum(features[s] for s in scales)/len(scales)
   with threadpool_limits(limits=2):scores=f@w.T+b
   m=classification_map(labels,scores);variants[name]=dict(scales=scales,val_map=float(m['map']),ap=dict(zip(CLASSES,m['ap'].tolist())))
   np.savez(run/f'{name}_predictions.npz',scores=scores,labels=labels,image_ids=np.array(ids));score_cache[name]=scores
   re=np.load(run/f'{name}_predictions.npz');assert classification_map(re['labels'],re['scores'])['map']==m['map']
   print('METRIC',name,variants[name],flush=True)
  previous=np.load(old/'mean480_576_neural_predictions.npz');np.testing.assert_allclose(previous['scores'],score_cache['dual'],rtol=0,atol=0)
  assert sha(checkpoint)==digest
  best=max(('dual','triple'),key=lambda k:variants[k]['val_map'])
  manifest=dict(variants[best],checkpoint=str(checkpoint),checkpoint_sha256=digest,classifier='checkpoint neural head',
    pooling=cfg['pooling'],class_order=list(CLASSES),selection_split='VOC2012 val; not independent test')
  dump(run/'recommended_pipeline.json',manifest)
  result=dict(status='completed',variants=variants,recommended=best,benchmark=timing,full_688_extraction_seconds=seconds,
   neural_updates=0,svm_fits=0,checkpoint_unchanged=True,dual_predictions_exact=True,feature_norms_verified=True,
   elapsed_seconds=time.perf_counter()-start)
  dump(run/'result.json',result)
 except Exception as e:
  dump(run/'result.json',dict(status='failed',error=repr(e)));raise
 print('COMPLETED',run,flush=True)
if __name__=='__main__':main()
