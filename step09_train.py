"""Small-set end-to-end FisherNet overfit gate; not full-dataset evaluation."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time
import numpy as np
import torch
from data.patch_inputs import VOCPatchDataset,patch_collate
from data.voc_classification import masked_bce
from models.fisher_classifier import FisherClassifier
from step04_train import subset_indices

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps',type=int,default=300)
    args=p.parse_args()
    if args.steps<1:p.error('positive steps required')
    run=ROOT/'runs'/f'step09_overfit_{datetime.now():%Y%m%d_%H%M%S}'
    run.mkdir(parents=True)
    def write(name,data):(run/name).write_text(json.dumps(data,indent=2),encoding='utf-8')
    history=[];result=dict(status='running',mode='train_subset_overfit')
    start=time.perf_counter()
    try:
        torch.manual_seed(42);torch.set_num_threads(2)
        torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        device='cuda' if torch.cuda.is_available() else 'cpu'
        gmm=ROOT/'runs/step08_gmm_20260920_014414_978178'
        model,_,_=FisherClassifier.from_gmm(ROOT/'artifacts/alexnet_baseline_v1',gmm,device)
        ds=VOCPatchDataset(ROOT/'datasets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012','train')
        indices=subset_indices(ds,32,42);samples=[ds[i] for i in indices]
        write('train_ids.json',[s['image_id'] for s in samples])
        groups={'backbone':model.encoder.patch_encoder,'fisher':model.encoder.fisher,'head':model.head}
        lrs={'backbone':3e-5,'fisher':1e-3,'head':1e-2}
        optimizer=torch.optim.Adam([dict(params=m.parameters(),lr=lrs[n]) for n,m in groups.items()])
        initial={n:next(m.parameters()).detach().clone() for n,m in groups.items()}
        config=dict(mode='overfit',samples=32,batch_size=2,max_steps=args.steps,seed=42,
                    optimizer='Adam',learning_rates=lrs,gradient_clip=10.,dropout=False,augmentation=False,
                    longest_side=480,all_grid_patches=True,gmm_run=str(gmm),
                    power_normalization='x/sqrt(abs(x)+1e-6)',loss='mean BCE over valid labels',
                    gate='same train subset BCE<0.05 and micro-F1>=0.95',device=device)
        write('config.json',config)
        def evaluate(step):
            model.eval();scores=[];labels=[];total=0.;count=0
            with torch.no_grad():
                for offset in range(0,32,2):
                    b=patch_collate(samples[offset:offset+2])
                    z=model([v.to(device) for v in b['images']])
                    y=b['targets'].to(device);mask=b['valid_labels'].to(device)
                    total+=float(masked_bce(z,y,mask))*int(mask.sum());count+=int(mask.sum())
                    scores.append(z.cpu());labels.append(b['raw_labels'])
            z=torch.cat(scores);raw=torch.cat(labels);valid=raw!=0;truth=raw==1;pred=z>=0
            tp=int((pred&truth&valid).sum());fp=int((pred&~truth&valid).sum());fn=int((~pred&truth&valid).sum())
            row=dict(step=step,bce=total/count,micro_f1=2*tp/max(2*tp+fp+fn,1),positive_recall=tp/max(tp+fn,1))
            history.append(row);write('history.json',history)
            np.savez(run/'train_predictions.npz',scores=z.numpy(),labels=raw.numpy(),image_ids=np.array([s['image_id'] for s in samples]))
            print(row,flush=True);return row
        evaluate(0)
        rng=np.random.default_rng(42);order=[]
        if device=='cuda':torch.cuda.reset_peak_memory_stats()
        for step in range(1,args.steps+1):
            if not order:order=rng.permutation(32).tolist()
            chosen=[order.pop(),order.pop()]
            b=patch_collate([samples[i] for i in chosen])
            # eval disables dropout, but leaves every parameter trainable.
            model.eval();optimizer.zero_grad(set_to_none=True)
            logits=model([v.to(device) for v in b['images']])
            loss=masked_bce(logits,b['targets'].to(device),b['valid_labels'].to(device))
            if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True)
            if step==1:
                result['first_gradient_norms']={n:sum(float(q.grad.norm()) for q in m.parameters() if q.grad is not None) for n,m in groups.items()}
                assert all(v>0 for v in result['first_gradient_norms'].values())
            optimizer.step()
            if step%25==0 or step==args.steps:
                row=evaluate(step)
                torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,
                                config=config,numpy_rng=rng.bit_generator.state,remaining_order=order,
                                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device=='cuda' else None),run/'last.pt')
                if row['bce']<.05 and row['micro_f1']>=.95:break
        changes={n:float((next(m.parameters()).detach()-initial[n]).abs().max()) for n,m in groups.items()}
        assert all(v>0 for v in changes.values())
        result.update(status='completed',steps=step,overfit_passed=row['bce']<.05 and row['micro_f1']>=.95,
                      initial=history[0],final=row,parameter_max_changes=changes,
                      peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2 if device=='cuda' else None,
                      elapsed_seconds=time.perf_counter()-start)
    except Exception as exc:
        result.update(status='failed',error=repr(exc));raise
    finally:
        write('result.json',result);print('RUN',run);print(json.dumps(result,indent=2))
        report=ROOT/'report/archive/run_details'/f'{run.name}.md'
        report.parent.mkdir(exist_ok=True,parents=True)
        report.write_text('# FisherNet小样本联合训练\n\n仅在train的32张图片上训练和检查拟合；不是val成绩。\n\n```json\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n```\n\n配置、历史、样本ID和checkpoint：'+str(run),encoding='utf-8')

if __name__=='__main__':main()
