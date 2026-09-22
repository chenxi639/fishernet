"""Matched 224 vs 320 whole-image resolution fine-tuning comparison."""
import argparse
from datetime import datetime
import gc
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from data import VOCClassification, baseline_transform, masked_bce, CLASSES
from models.alexnet_baseline import AlexNetBaseline
from step04_train import evaluate

ROOT=Path(__file__).resolve().parent
WEAK=('bottle','sofa','pottedplant')


def dump(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def write_report(run,cfg,result,history):
    rows=['# 整图分辨率针对性实验', '', '## 目的与控制条件', '',
          f'- image_size={cfg["image_size"]}；起点 `{cfg["source"]}`。',
          '- train=5717，val=5823；恢复同一模型和SGD动量，两组使用相同种子、顺序、翻转及学习率。',
          '- 不新增网络模块，使用AlexNet原有池化。单张整图单尺度输入，不做patch/SPP或多尺度融合。',
          '- 旧层/新层LR=0.0001/0.001，前两轮后乘0.1；batch=16、SGD momentum=0.9、weight_decay=0.0005，masked mean BCE。',
          '- 最佳checkpoint按完整val整体mAP选择；不能拼接各类各轮最高值。',
          '- 新分辨率需要重新评估起点；相同权重在224/320的指标不必相同。', '',
          '## 指标', '', '|相对轮次|在线训练BCE|val BCE|val mAP %|bottle AP %|sofa AP %|pottedplant AP %|',
          '|---|---:|---:|---:|---:|---:|---:|']
    for row in history:
        m=row['metrics']; loss=f'{row["train_bce"]:.5f}' if row.get('train_bce') is not None else '—'
        rows.append(f'|{row["epoch"]}|{loss}|{m["loss"]:.5f}|{m["map"]*100:.2f}|'+
                    '|'.join(f'{m["ap"][c]*100:.2f}' for c in WEAK)+'|')
    rows += ['', '## 结果、限制与改进方向', '',
             f'- 状态：{result.get("status")}；最佳相对轮次：{result.get("best_epoch")}；峰值已分配显存：{result.get("peak_allocated_mib",0):.1f} MiB。',
             f'- 耗时：{result.get("elapsed_seconds",0):.1f}秒。',
             '- 仅单个随机种子的有限轮次对照，val用于选模，结果存在选择偏差。弱类别范围由前轮结果确定。',
             '- 增加输入分辨率也增加计算量；最终与224控制组和原基线共同比较，只有收益成立才推荐升级。',
             '- 若瓶子提升仍有限，进一步分析上下文误报与外观差异，避免盲目继续增大分辨率。',
             f'- 原始配置、每轮预测/权重、结果：`runs/{run.name}`。',
             '', '## 复现命令', '', '```powershell',cfg['command'],'```','']
    detail_dir=ROOT/'report'/'archive'/'run_details'
    detail_dir.mkdir(parents=True,exist_ok=True)
    (detail_dir/f'{run.name}.md').write_text('\n'.join(rows),encoding='utf-8')


def train_arm(source,size,epochs,workers):
    run=ROOT/'runs'/f'step04_targeted_{size}_{datetime.now():%Y%m%d_%H%M%S}'
    run.mkdir(parents=True)
    cfg=dict(source=str(source.resolve()),image_size=size,epochs=epochs,workers=workers,seed=2718,
             command=f'python step04_targeted.py --source "{source}" --sizes {size} --epochs {epochs} --workers {workers}',
             source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),torch=torch.__version__,
             gpu=torch.cuda.get_device_name(0),batch_size=16,lr=[.0001,.001],gamma=.1,decay_after=2)
    dump(run/'config.json',cfg)
    result,history={'status':'running'},[]
    started=time.perf_counter()
    try:
        torch.manual_seed(2718)
        model=AlexNetBaseline().cuda()
        checkpoint=torch.load(source,map_location='cpu',weights_only=False)
        if tuple(checkpoint['classes'])!=CLASSES: raise ValueError('class mismatch')
        model.load_state_dict(checkpoint['model'])
        optimizer=torch.optim.SGD(model.optimizer_groups(),momentum=.9,weight_decay=.0005)
        optimizer.load_state_dict(checkpoint['optimizer'])
        del checkpoint
        root=ROOT/'datasets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012'
        train=VOCClassification(root,'train',baseline_transform(True,size))
        val=VOCClassification(root,'val',baseline_transform(False,size))
        dump(run/'train_ids.json',train.ids); dump(run/'val_ids.json',val.ids)
        ev=DataLoader(val,batch_size=16,num_workers=workers,persistent_workers=workers>0)
        initial,arrays=evaluate(model,ev,'cuda')
        history.append(dict(epoch=0,metrics=initial,train_bce=None))
        best,best_epoch,best_metrics=initial['map'],0,initial
        def save(path,epoch):
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),config=cfg,
                            classes=CLASSES,epoch=epoch),path)
        save(run/'best.pt',0)
        np.savez_compressed(run/'best_predictions.npz',**arrays)
        torch.cuda.reset_peak_memory_stats()
        print('INITIAL',size,json.dumps(initial),flush=True)
        for epoch in range(1,epochs+1):
            torch.manual_seed(2718+epoch); random.seed(2718+epoch); np.random.seed(2718+epoch)
            lrs=[.0001,.001] if epoch<=2 else [.00001,.0001]
            for group,lr in zip(optimizer.param_groups,lrs): group['lr']=lr
            loader=DataLoader(train,batch_size=16,shuffle=True,num_workers=workers,pin_memory=True,
                              generator=torch.Generator().manual_seed(2718+epoch))
            model.train(); total,count=0.,0
            for i,batch in enumerate(loader):
                x,y,m=[batch[k].cuda() for k in ('image','target','valid')]
                optimizer.zero_grad(set_to_none=True)
                loss=masked_bce(model(x),y,m)
                if not torch.isfinite(loss): raise RuntimeError('nonfinite loss')
                loss.backward(); optimizer.step()
                n=int(m.sum()); total+=loss.item()*n; count+=n
                if (i+1)%150==0: print(f'size={size} epoch={epoch} batch={i+1} BCE={total/count:.5f}',flush=True)
            metrics,arrays=evaluate(model,ev,'cuda')
            history.append(dict(epoch=epoch,metrics=metrics,train_bce=total/count,lrs=lrs))
            if metrics['map']>best:
                best,best_epoch,best_metrics=metrics['map'],epoch,metrics
                save(run/'best.pt',epoch)
                np.savez_compressed(run/'best_predictions.npz',**arrays)
            save(run/'last.pt',epoch)
            np.savez_compressed(run/'final_predictions.npz',**arrays)
            result.update(best_map=best,best_epoch=best_epoch,best_metrics=best_metrics,last_metrics=metrics,
                          elapsed_seconds=time.perf_counter()-started,peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2)
            dump(run/'history.json',history); dump(run/'result.json',result)
            write_report(run,cfg,result,history)
            print('EPOCH',size,epoch,json.dumps(metrics),flush=True)
        result['status']='completed'
    except Exception as exc:
        result.update(status='failed',error=repr(exc)); raise
    finally:
        result['elapsed_seconds']=time.perf_counter()-started
        dump(run/'history.json',history); dump(run/'result.json',result)
        write_report(run,cfg,result,history)
        print('RUN',run,flush=True)
    return str(run)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--sizes',type=int,nargs='+',default=[224,320])
    p.add_argument('--epochs',type=int,default=4)
    p.add_argument('--workers',type=int,default=2)
    args=p.parse_args()
    if args.epochs<1 or args.workers<0 or any(s not in (224,320) for s in args.sizes):
        p.error('positive epochs, nonnegative workers, supported sizes 224/320')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    torch.set_num_threads(2)
    for size in args.sizes:
        train_arm(args.source,size,args.epochs,args.workers)
        gc.collect(); torch.cuda.empty_cache()
