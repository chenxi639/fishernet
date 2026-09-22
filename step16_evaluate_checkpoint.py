"""Evaluate a new trained checkpoint with fixed 480/576/688 Fisher means."""
import argparse,json,time,shutil
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from step13_evaluate import sha,dump
from data.patch_inputs import VOCPatchDataset
from data.voc_classification import CLASSES
from models.fisher_classifier import FisherClassifier,normalize_fisher
from metrics import classification_map
ROOT=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True,type=Path);a=p.parse_args()
 run=a.checkpoint.parent/'triple_evaluation';run.mkdir(exist_ok=True);start=time.perf_counter()
 cfg=dict(checkpoint=str(a.checkpoint.resolve()),checkpoint_sha256=sha(a.checkpoint),scales=[480,576,688],
  normalization='per-scale smooth power (eps1e-6) then L2',pooling='arithmetic mean; no post-mean normalization',
  classifier='unchanged checkpoint neural head',split='VOC2012 val5823',batch_size=2,neural_updates=0)
 if (run/'config.json').exists():assert json.loads((run/'config.json').read_text())==cfg
 else:dump(run/'config.json',cfg)
 print('RUN',run,flush=True);dump(run/'result.json',dict(status='running'))
 try:
  torch.set_num_threads(2);torch.manual_seed(42);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
  device='cuda' if torch.cuda.is_available() else 'cpu'
  model,_,_=FisherClassifier.from_gmm(ROOT/'artifacts/alexnet_baseline_v1',ROOT/'runs/step08_gmm_20260920_014414_978178',device)
  ck=torch.load(a.checkpoint,map_location=device,weights_only=False);model.load_state_dict(ck['model']);ids=ck['val_ids'];epoch=ck['epoch']
  assert len(ids)==5823 and not set(ids)&set(ck['train_ids']);del ck
  model.eval();model.requires_grad_(False)
  w=model.head.weight.detach().cpu().numpy().copy();b=model.head.bias.detach().cpu().numpy().copy()
  features={};times={}
  for scale in cfg['scales']:
   ds=VOCPatchDataset(ROOT/'datasets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012','val',scale);assert ds.ids==ids
   labels=ds.raw_labels.numpy().copy();file=run/f'val_{scale}.npy';marker=run/f'val_{scale}_complete.json'
   if marker.exists():
    info=json.loads(marker.read_text());assert sha(file)==info['sha256']
   else:
    t=time.perf_counter();x=np.lib.format.open_memmap(file,mode='w+',dtype=np.float32,shape=(len(ds),16384))
    with torch.inference_mode():
     for j in range(0,len(ds),2):
      images=[ds[i]['image'].to(device) for i in range(j,min(j+2,len(ds)))]
      f=normalize_fisher(model.encoder(images)['fisher']);assert torch.isfinite(f).all()
      if j==0:torch.testing.assert_close(model(images),model.head(f),rtol=0,atol=0)
      x[j:j+len(images)]=f.cpu().numpy()
      if (j+2)%512==0 or j+2>=len(ds):
       status=dict(scale=scale,processed=min(j+2,len(ds)),total=len(ds),elapsed_seconds=time.perf_counter()-start)
       dump(run/'progress.json',status);print('PROGRESS',status,flush=True)
    x.flush();del x;info=dict(sha256=sha(file),seconds=time.perf_counter()-t);dump(marker,info)
   times[str(scale)]=info['seconds'];features[scale]=np.load(file,mmap_mode='r')
   assert features[scale].shape==(5823,16384)
   for j in range(0,len(ds),128):np.testing.assert_allclose(np.linalg.norm(features[scale][j:j+128],axis=1),1,atol=2e-6)
  avg=(features[480]+features[576]+features[688])/3
  with threadpool_limits(limits=2):scores=avg@w.T+b;single=features[480]@w.T+b
  saved_single=np.load(a.checkpoint.parent/'last_predictions.npz')
  assert saved_single['image_ids'].tolist()==ids;np.testing.assert_array_equal(saved_single['labels'],labels)
  np.testing.assert_allclose(saved_single['scores'],single,rtol=2e-4,atol=2e-5)
  ref=np.load(ROOT/'runs/step14_688_20260921_172409/triple_predictions.npz')
  assert ref['image_ids'].tolist()==ids;np.testing.assert_array_equal(ref['labels'],labels)
  m=classification_map(labels,scores);old=classification_map(ref['labels'],ref['scores'])
  valid=labels!=0;truth=labels==1;pred=scores>=0
  tp=int((truth&pred).sum());fp=int(((labels==-1)&pred).sum());fn=int((truth&~pred).sum())
  z=torch.from_numpy(scores);target=torch.from_numpy(truth.astype(np.float32));mask=torch.from_numpy(valid)
  bce=float(torch.nn.functional.binary_cross_entropy_with_logits(z[mask],target[mask]))
  diag={}
  for name in ('bottle','pottedplant','sofa'):
   k=CLASSES.index(name);diag[name]=dict(tp=int((truth[:,k]&pred[:,k]).sum()),fp=int(((labels[:,k]==-1)&pred[:,k]).sum()),fn=int((truth[:,k]&~pred[:,k]).sum()))
  np.savez(run/'predictions.npz',scores=scores,labels=labels,image_ids=np.array(ids))
  reread=np.load(run/'predictions.npz');assert classification_map(reread['labels'],reread['scores'])['map']==m['map']
  assert sha(a.checkpoint)==cfg['checkpoint_sha256']
  improved=m['map']>old['map']
  manifest=dict(checkpoint=str(a.checkpoint.resolve()),checkpoint_sha256=cfg['checkpoint_sha256'],scales=cfg['scales'],
   classifier='checkpoint neural head',normalization=cfg['normalization'],pooling=cfg['pooling'],class_order=list(CLASSES),val_map=float(m['map']),selection_split='VOC2012 val; not independent test')
  dump(run/'candidate_pipeline.json',manifest)
  if improved:dump(run/'practical_pipeline.json',manifest)
  else:shutil.copy2(ROOT/'runs/step15_864_20260921_205024/practical_pipeline.json',run/'practical_pipeline.json')
  result=dict(status='completed',epoch=epoch,val_map=float(m['map']),val_bce=bce,ap=dict(zip(CLASSES,m['ap'].tolist())),
   micro_f1=2*tp/(2*tp+fp+fn),positive_recall=tp/(tp+fn),weak_class_counts=diag,reference_map=float(old['map']),
   reference_ap=dict(zip(CLASSES,old['ap'].tolist())),delta_map_pp=float((m['map']-old['map'])*100),candidate_improved=bool(improved),
   extraction_seconds=times,elapsed_seconds=time.perf_counter()-start,single480_reproduced=True,prediction_ap_recomputed=True,checkpoint_unchanged=True)
  dump(run/'result.json',result);print('COMPLETED',json.dumps(result),flush=True)
 except Exception as e:dump(run/'result.json',dict(status='failed',error=repr(e)));raise
if __name__=='__main__':main()
