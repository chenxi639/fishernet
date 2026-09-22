"""Score existing epoch6 training features; no feature extraction or fitting."""
import json
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from metrics import classification_map
from data.voc_classification import CLASSES
from step13_evaluate import sha,dump
ROOT=Path(__file__).resolve().parent
def main():
 run=ROOT/'runs/step17_diagnosis_20260922_123454';cache=ROOT/'runs/step13_eval_20260921_170016'
 cfg=json.loads((cache/'config.json').read_text());checkpoint=Path(cfg['checkpoint']);assert sha(checkpoint)==cfg['checkpoint_sha256']
 torch.set_num_threads(2);ck=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
 w=ck['model']['head.weight'].numpy();b=ck['model']['head.bias'].numpy();assert ck['epoch']==6
 arrays=[]
 for scale in (480,576):
  p=cache/f'train_{scale}.npy';assert sha(p)==json.loads((cache/f'train_{scale}_complete.json').read_text())['sha256']
  arrays.append(np.load(p,mmap_mode='r'))
 x=(arrays[0]+arrays[1])*.5
 with threadpool_limits(limits=2):scores=x@w.T+b
 y=np.load(cache/'train_labels.npy');ids=json.loads((cache/'train_ids.json').read_text());assert ids==ck['train_ids']
 val=np.load(cache/'mean480_576_neural_predictions.npz');assert not set(ids)&set(val['image_ids'])
 metrics={}
 for split,labels,logits in [('train',y,scores),('val',val['labels'],val['scores'])]:
  m=classification_map(labels,logits);mask=torch.from_numpy(labels!=0);target=torch.from_numpy((labels==1).astype(np.float32));z=torch.from_numpy(logits)
  metrics[split]=dict(map=float(m['map']),ap=dict(zip(CLASSES,m['ap'].tolist())),bce=float(torch.nn.functional.binary_cross_entropy_with_logits(z[mask],target[mask])))
 pool=[];context={}
 for name in ('bottle','pottedplant','sofa'):
  k=CLASSES.index(name);context[name]={}
  for typ,label in [('low_score_positive',1),('high_score_negative',-1)]:
   ii=np.flatnonzero(y[:,k]==label);order=ii[np.argsort(scores[ii,k],kind='stable')]
   if label==-1:order=order[::-1]
   for i in order[:16]:pool.append(dict(id=ids[i],target=name,kind=typ,score=float(scores[i,k]),raw_label=int(y[i,k]),positive_classes=[CLASSES[j] for j in range(20) if y[i,j]==1]))
  for other in ('chair','diningtable'):
   if other==name:continue
   j=CLASSES.index(other);mask=(y[:,k]==-1)&(y[:,j]==1)
   context[name][other]=dict(valid_negative_count=int(mask.sum()),false_positives=int((mask&(scores[:,k]>=0)).sum()))
 out=dict(status='passed',epoch=6,scales=[480,576],feature_source=str(cache),checkpoint_sha256=cfg['checkpoint_sha256'],
  neural_updates=0,cnn_forward_calls=0,metrics=metrics,training_context=context,
  candidate_rows=len(pool),candidate_unique_images=len({v['id'] for v in pool}),
  caveats=['Cached epoch6 features scored by epoch6 head; not epoch7 training predictions.',
   'Training metrics are fit diagnostics, not generalization estimates.',
   'Candidates selected exclusively from train; manual annotation-boundary review required before sampling changes.'])
 dump(run/'training_fit_audit.json',out);dump(run/'training_review_candidates.json',pool)
 print(json.dumps(out,indent=2))
if __name__=='__main__':main()
