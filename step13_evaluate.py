"""Frozen Fisher features: scale pooling and train-only one-vs-rest SVM."""
import argparse, hashlib, json, time, warnings
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from sklearn.svm import LinearSVC
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from data.patch_inputs import VOCPatchDataset
from data.voc_classification import CLASSES
from models.fisher_classifier import FisherClassifier, normalize_fisher
from metrics import classification_map

ROOT=Path(__file__).resolve().parent
def dump(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def fit_svm(x,y):
 weights=[];bias=[];iterations=[]
 for k,name in enumerate(CLASSES):
  valid=y[:,k]!=0
  assert set(y[valid,k])=={-1,1}
  clf=LinearSVC(C=1.,penalty='l2',loss='squared_hinge',dual=True,tol=1e-4,max_iter=10000,random_state=42)
  with warnings.catch_warnings():
   warnings.simplefilter('error',ConvergenceWarning)
   clf.fit(x[valid],y[valid,k])
  weights.append(clf.coef_[0]);bias.append(clf.intercept_[0]);iterations.append(int(clf.n_iter_))
  print('SVM',name,'iterations',clf.n_iter_,flush=True)
 return np.stack(weights),np.array(bias),iterations
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--checkpoint',type=Path,default=ROOT/'runs/step12_scale_20260921_130318/best.pt')
 p.add_argument('--run',type=Path,help='reuse completed feature caches from this run')
 a=p.parse_args();run=a.run or ROOT/'runs'/f'step13_eval_{datetime.now():%Y%m%d_%H%M%S}'
 run.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
 print('RUN',run,flush=True)
 cfg=dict(checkpoint=str(a.checkpoint.resolve()),checkpoint_sha256=sha(a.checkpoint),scales=[480,576],
  pooling='mean of per-scale smooth-power/L2 normalized FV; no post-mean normalization',
  svm=dict(C=1,loss='squared_hinge',penalty='l2',dual=True,tol=1e-4,max_iter=10000,fit_intercept=True,intercept_scaling=1,random_state=42),
  neural_updates=0,augmentation='none; eval mode',split='VOC2012 train5717 -> val5823',normalization_epsilon=1e-6)
 if (run/'config.json').exists():assert json.loads((run/'config.json').read_text())==cfg
 else:dump(run/'config.json',cfg)
 torch.set_num_threads(2);torch.manual_seed(42)
 torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 device='cuda' if torch.cuda.is_available() else 'cpu'
 model,_,_=FisherClassifier.from_gmm(ROOT/'artifacts/alexnet_baseline_v1',ROOT/'runs/step08_gmm_20260920_014414_978178',device)
 ck=torch.load(a.checkpoint,map_location=device,weights_only=False);model.load_state_dict(ck['model']);model.eval();model.requires_grad_(False)
 weight=model.head.weight.detach().cpu().numpy().copy();bias=model.head.bias.detach().cpu().numpy().copy()
 ids={s:ck[s+'_ids'] for s in ('train','val')};del ck
 assert not set(ids['train'])&set(ids['val'])
 data=ROOT/'datasets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012'
 arrays={};labels={};cache_info={}
 for split in ('val','train'):
  arrays[split]={}
  for scale in (480,576):
   ds=VOCPatchDataset(data,split,scale);assert ds.ids==ids[split]
   labels[split]=ds.raw_labels.numpy().copy();np.save(run/f'{split}_labels.npy',labels[split]);dump(run/f'{split}_ids.json',ds.ids)
   path=run/f'{split}_{scale}.npy';marker=run/f'{split}_{scale}_complete.json'
   if marker.exists():
    info=json.loads(marker.read_text());assert sha(path)==info['sha256'];cache_info[f'{split}_{scale}']=info
   else:
    x=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(len(ds),16384))
    then=time.perf_counter()
    with torch.inference_mode():
     for j in range(0,len(ds),2):
      images=[ds[i]['image'].to(device) for i in range(j,min(j+2,len(ds)))]
      f=normalize_fisher(model.encoder(images)['fisher'])
      assert torch.isfinite(f).all()
      if j==0:
       torch.testing.assert_close(model(images),model.head(f),rtol=0,atol=0)
      x[j:j+len(images)]=f.cpu().numpy()
      if (j+2)%512==0 or j+2>=len(ds):
       progress=dict(phase='extract',split=split,scale=scale,processed=min(j+2,len(ds)),total=len(ds),elapsed_seconds=time.perf_counter()-start)
       dump(run/'progress.json',progress);print('PROGRESS',progress,flush=True)
    x.flush();del x
    info=dict(sha256=sha(path),seconds=time.perf_counter()-then,shape=[len(ds),16384]);dump(marker,info);cache_info[f'{split}_{scale}']=info
   arrays[split][scale]=np.load(path,mmap_mode='r')
 variants={};saved={}
 for name,scales in [('single480',[480]),('mean480_576',[480,576])]:
  features={s:np.asarray(arrays[s][480]) if len(scales)==1 else (arrays[s][480]+arrays[s][576])*.5 for s in ('train','val')}
  for head in ('neural','svm'):
   key=name+'_'+head;then=time.perf_counter()
   if head=='svm':
    with threadpool_limits(limits=2):w,b,it=fit_svm(features['train'],labels['train'])
    np.savez(run/f'{key}_model.npz',weight=w,bias=b,classes=np.array(CLASSES));saved[key]=dict(iterations=it,fit_seconds=time.perf_counter()-then)
   else:w,b=weight,bias
   with threadpool_limits(limits=2):scores=features['val']@w.T+b
   m=classification_map(labels['val'],scores)
   variants[key]=dict(val_map=float(m['map']),ap=dict(zip(CLASSES,m['ap'].tolist())))
   np.savez(run/f'{key}_predictions.npz',scores=scores,labels=labels['val'],image_ids=np.array(ids['val']))
   dump(run/'partial_metrics.json',variants);print('METRIC',key,variants[key],flush=True)
 reference=np.load(a.checkpoint.parent/'best_predictions.npz')
 single=np.load(run/'single480_neural_predictions.npz')
 assert np.array_equal(reference['image_ids'],single['image_ids'])
 np.testing.assert_allclose(reference['scores'],single['scores'],rtol=2e-4,atol=2e-5)
 assert abs(classification_map(reference['labels'],reference['scores'])['map']-variants['single480_neural']['val_map'])<1e-5
 assert sha(a.checkpoint)==cfg['checkpoint_sha256']
 result=dict(status='completed',variants=variants,svm=saved,cache=cache_info,neural_updates=0,checkpoint_unchanged=True,
  reference_logits_max_error=float(np.max(np.abs(reference['scores']-single['scores']))),elapsed_seconds=time.perf_counter()-start)
 dump(run/'result.json',result);print('COMPLETED',run,flush=True)
if __name__=='__main__':main()
