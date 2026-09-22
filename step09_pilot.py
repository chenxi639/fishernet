"""Joint FisherNet training with independent validation; 0 images selects full split."""
import argparse
from datetime import datetime
import json
import math
import hashlib
from pathlib import Path
import time
import numpy as np
import torch
from data.patch_inputs import VOCPatchDataset,patch_collate
from data.voc_classification import masked_bce,CLASSES
from models.fisher_classifier import FisherClassifier
from metrics import classification_map
from step04_train import subset_indices

ROOT=Path(__file__).resolve().parent

def scale_learning_rates(optimizer,factor):
    """Scale restored rates once, preserving Adam moments and step counters."""
    if not math.isfinite(factor) or factor<=0:raise ValueError('finite positive LR scale required')
    before=[g['lr'] for g in optimizer.param_groups]
    for g in optimizer.param_groups:g['lr']*=factor
    return before,[g['lr'] for g in optimizer.param_groups]

def set_group_weight_decays(optimizer,values):
    """Apply one explicit weight decay per backbone/Fisher/head parameter group."""
    values=[float(v) for v in values]
    if len(values)!=len(optimizer.param_groups) or any(not math.isfinite(v) or v<0 for v in values):
        raise ValueError('one finite nonnegative weight decay is required per parameter group')
    for group,value in zip(optimizer.param_groups,values):group['weight_decay']=value
    return [g['weight_decay'] for g in optimizer.param_groups]

def make_optimizer(param_groups,name):
    """Build the requested optimizer; group learning rates are already attached."""
    if name=='adam':return torch.optim.Adam(param_groups)
    if name=='adamw':return torch.optim.AdamW(param_groups,weight_decay=0.)
    raise ValueError(f'unsupported optimizer: {name}')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-images',type=int,default=256)
    p.add_argument('--val-images',type=int,default=256)
    p.add_argument('--epochs',type=int,default=2)
    p.add_argument('--resume',type=Path)
    p.add_argument('--lr-scale',type=float,default=1.,help='multiply restored optimizer rates once; default keeps rates')
    p.add_argument('--train-scales',type=int,nargs='+',help='uniform random longest sides for training only; validation stays 480')
    p.add_argument('--optimizer',choices=('adam','adamw'),default='adam',help='optimizer algorithm; explicit on resume')
    p.add_argument('--weight-decays',type=float,nargs=3,metavar=('BACKBONE','FISHER','HEAD'),default=(0.,0.,0.),help='per-group decay in backbone/Fisher/head order')
    args=p.parse_args()
    if args.train_scales and (min(args.train_scales)<480 or len(set(args.train_scales))!=len(args.train_scales)):p.error('unique training scales >=480 required')
    if min(args.train_images,args.val_images)<0 or args.epochs<1:p.error('nonnegative counts, positive epochs required')
    if not math.isfinite(args.lr_scale) or args.lr_scale<=0:p.error('finite positive --lr-scale required')
    if args.lr_scale!=1. and not args.resume:p.error('--lr-scale requires --resume')
    if any(not math.isfinite(v) or v<0 for v in args.weight_decays):p.error('finite nonnegative --weight-decays required')
    if args.optimizer!='adamw' and any(args.weight_decays):p.error('nonzero --weight-decays require --optimizer adamw for this controlled branch')
    prefix='step10_full' if args.train_images==0 and args.val_images==0 else 'step09_pilot'
    if args.lr_scale!=1.:prefix='step11_lrbranch'
    if args.train_scales:prefix='step12_scale'
    if args.optimizer=='adamw' or any(args.weight_decays):prefix='step18_regularization'
    run=ROOT/'runs'/f'{prefix}_{datetime.now():%Y%m%d_%H%M%S}'
    run.mkdir(parents=True)
    def write(name,obj):(run/name).write_text(json.dumps(obj,indent=2),encoding='utf-8')
    result=dict(status='running');history=[];start=time.perf_counter()
    try:
        torch.manual_seed(42);torch.set_num_threads(2)
        torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        device='cuda' if torch.cuda.is_available() else 'cpu'
        model,_,_=FisherClassifier.from_gmm(ROOT/'artifacts/alexnet_baseline_v1',ROOT/'runs/step08_gmm_20260920_014414_978178',device)
        root=ROOT/'datasets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012'
        train=VOCPatchDataset(root,'train');val=VOCPatchDataset(root,'val')
        groups={'backbone':model.encoder.patch_encoder,'fisher':model.encoder.fisher,'head':model.head}
        lrs={'backbone':1e-5,'fisher':3e-4,'head':3e-3}
        opt=make_optimizer([dict(params=m.parameters(),lr=lrs[n]) for n,m in groups.items()],args.optimizer)
        rng=np.random.default_rng(42);first_epoch=1;best=-1.
        if args.resume:
            ck=torch.load(args.resume,map_location=device,weights_only=False)
            cfg=ck['config'];ti=[train.ids.index(i) for i in ck['train_ids']];vi=[val.ids.index(i) for i in ck['val_ids']]
            model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
            rng.bit_generator.state=ck['numpy_rng'];torch.set_rng_state(ck['torch_rng'].cpu())
            if device=='cuda':torch.cuda.set_rng_state_all([v.cpu() for v in ck['cuda_rng']])
            first_epoch=ck['epoch']+1;best=ck['best_map'];cfg=dict(cfg,resumed_from=str(args.resume))
            if args.epochs<first_epoch:raise ValueError('--epochs must exceed resumed epoch')
        else:
            if args.train_images>len(train) or args.val_images>len(val):raise ValueError('requested subset exceeds split')
            ti=list(range(len(train))) if not args.train_images else subset_indices(train,args.train_images,42)
            vi=list(range(len(val))) if not args.val_images else subset_indices(val,args.val_images,43)
            cfg=dict(train_images=len(ti),val_images=len(vi),batch_size=2,seed=42,optimizer='Adam',
                     learning_rates=lrs,clip_norm=10.,dropout=True,random_horizontal_flip=True,
                     longest_side=480,subset_policy='all classes covered then seeded fill, unless full split',
                     schedule='constant learning rate',normalization_epsilon=1e-6,
                     initialization='baseline_v1 + step08 GMM + random head; not tiny-overfit checkpoint')
        prior_optimizer=cfg.get('optimizer','Adam')
        cfg['optimizer']=type(opt).__name__
        cfg['optimizer_transition']=dict(from_checkpoint=prior_optimizer,to=type(opt).__name__) if args.resume else None
        cfg['weight_decays']=dict(zip(groups,set_group_weight_decays(opt,args.weight_decays)))
        scales=args.train_scales or cfg.get('train_scales',[480])
        train_by_scale={s:VOCPatchDataset(root,'train',longest_side=s) for s in scales}
        assert all(d.ids==train.ids for d in train_by_scale.values())
        cfg['train_scales']=scales
        cfg['validation_longest_side']=480
        cfg['scale_sampling']='uniform per image; independent numpy RNG seed 42 + epoch * 100003'
        before_lr,after_lr=scale_learning_rates(opt,args.lr_scale)
        cfg['learning_rates']=dict(zip(groups,after_lr))
        cfg['applied_lr_scale']=args.lr_scale
        if args.lr_scale!=1.:
            events=list(cfg.get('learning_rate_events',[]))
            events.append(dict(after_epoch=first_epoch-1,factor=args.lr_scale,before=before_lr,after=after_lr))
            cfg['learning_rate_events']=events
            cfg['schedule']='constant within run; restored rates scaled once at branch start'
        if args.resume:
            digest=hashlib.sha256()
            with args.resume.open('rb') as source:
                for block in iter(lambda:source.read(1024*1024),b''):digest.update(block)
            cfg['resume_checkpoint_sha256']=digest.hexdigest()
            cfg['resume_epoch']=first_epoch-1
        cfg['target_epochs']=args.epochs
        cfg['evaluation_scope']='full VOC2012 val' if len(vi)==len(val) else 'VOC2012 val subset'
        train_ids=[train.ids[i] for i in ti];val_ids=[val.ids[i] for i in vi]
        assert not set(train_ids)&set(val_ids)
        write('config.json',cfg);write('train_ids.json',train_ids);write('val_ids.json',val_ids)
        print('RUN',run,flush=True)
        def progress(phase,epoch,processed,total,**extra):
            status=dict(phase=phase,epoch=epoch,processed=processed,total=total,
                        elapsed_seconds=time.perf_counter()-start,**extra)
            write('progress.json',status)
            print('PROGRESS',status,flush=True)
        def evaluate(epoch):
            model.eval();scores=[];labels=[];total=0.;count=0
            with torch.no_grad():
                for j in range(0,len(vi),2):
                    b=patch_collate([val[i] for i in vi[j:j+2]])
                    z=model([v.to(device) for v in b['images']])
                    y=b['targets'].to(device);mask=b['valid_labels'].to(device)
                    if not torch.isfinite(z).all():raise RuntimeError('nonfinite validation logits')
                    total+=float(masked_bce(z,y,mask))*int(mask.sum());count+=int(mask.sum())
                    scores.append(z.cpu().numpy());labels.append(b['raw_labels'].numpy())
                    if (j+2)%512==0 or j+2>=len(vi):progress('validation',epoch,min(j+2,len(vi)),len(vi))
            scores=np.concatenate(scores);labels=np.concatenate(labels)
            ap=classification_map(labels,scores);valid=labels!=0;truth=labels==1;pred=scores>=0
            tp=int((pred&truth&valid).sum());fp=int((pred&~truth&valid).sum());fn=int((~pred&truth&valid).sum())
            m=dict(epoch=epoch,val_bce=total/count,val_map=float(ap['map']),
                   micro_f1=2*tp/max(2*tp+fp+fn,1),positive_recall=tp/max(tp+fn,1),
                   ap=dict(zip(CLASSES,ap['ap'].tolist())))
            return m,dict(scores=scores,labels=labels,image_ids=np.array(val_ids))
        initial,_=evaluate(first_epoch-1);history.append(initial);write('history.json',history)
        print('INITIAL',initial,flush=True)
        if device=='cuda':torch.cuda.reset_peak_memory_stats()
        updates=0
        for epoch in range(first_epoch,args.epochs+1):
            order=rng.permutation(ti);model.train();total=0.;count=0
            scale_rng=np.random.default_rng(42+epoch*100003)
            scale_counts={str(s):0 for s in scales}
            scale_choices=[]
            for j in range(0,len(order),2):
                chosen=scale_rng.choice(scales,size=len(order[j:j+2])).tolist()
                for s in chosen:scale_counts[str(s)]+=1
                scale_choices.extend(chosen)
                b=patch_collate([train_by_scale[s][int(i)] for i,s in zip(order[j:j+2],chosen)])
                images=[(torch.flip(v,[-1]) if rng.random()<.5 else v).to(device) for v in b['images']]
                opt.zero_grad(set_to_none=True);z=model(images)
                mask=b['valid_labels'].to(device);loss=masked_bce(z,b['targets'].to(device),mask)
                if not torch.isfinite(loss):raise RuntimeError('nonfinite training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True);opt.step()
                updates+=1;total+=float(loss.detach())*int(mask.sum());count+=int(mask.sum())
                if (j+2)%256==0 or j+2>=len(order):
                    progress('training',epoch,min(j+2,len(order)),len(order),online_bce=total/count)
            write(f'epoch{epoch}_scale_assignment.json',dict(image_ids=[train.ids[int(i)] for i in order],longest_sides=scale_choices,counts=scale_counts))
            row,pred=evaluate(epoch);row['train_scale_counts']=scale_counts;row['train_bce_online']=total/count;history.append(row);write('history.json',history)
            np.savez(run/'last_predictions.npz',**pred)
            improved=row['val_map']>best
            if improved:best=row['val_map']
            state=dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,best_map=best,config=cfg,
                       train_ids=train_ids,val_ids=val_ids,numpy_rng=rng.bit_generator.state,
                       torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device=='cuda' else None)
            torch.save(state,run/'last.pt')
            if improved:
                torch.save(state,run/'best.pt');np.savez(run/'best_predictions.npz',**pred)
            print('EPOCH',row,flush=True)
        # Test exact checkpoint reload at logits level (same eval inputs/device).
        probe=[val[vi[0]]['image'].to(device)]
        with torch.no_grad():before=model(probe).clone()
        loaded=torch.load(run/'last.pt',map_location=device,weights_only=False)
        model.load_state_dict(loaded['model']);opt.load_state_dict(loaded['optimizer'])
        with torch.no_grad():after=model(probe)
        torch.testing.assert_close(before,after,rtol=0,atol=0)
        result.update(status='completed',initial=initial,final=row,best_val_map=best,updates=updates,
                      train_images=len(ti),val_images=len(vi),checkpoint_reload_exact=True,
                      peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2 if device=='cuda' else None,
                      elapsed_seconds=time.perf_counter()-start)
    except Exception as exc:
        result.update(status='failed',error=repr(exc));raise
    finally:
        write('result.json',result);print('RUN',run);print(json.dumps(result,indent=2))
        out=ROOT/'report/archive/run_details'/f'{run.name}.md';out.parent.mkdir(exist_ok=True,parents=True)
        out.write_text('# FisherNet联合训练\n\n评估范围见config与样本ID；val用于选模，不是论文test成绩。\n\n```json\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n```\n\n配置、样本、逐轮历史与checkpoint：'+str(run),encoding='utf-8')

if __name__=='__main__':main()
