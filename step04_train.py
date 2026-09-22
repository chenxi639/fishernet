"""Small-set overfit / short whole-image baseline. Writes report on every run."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from data import VOCClassification, CLASSES, baseline_transform, masked_bce
from metrics import classification_map
from models.alexnet_baseline import AlexNetBaseline

ROOT = Path(__file__).resolve().parent


def evaluate(model, loader, device):
    model.eval()
    total, count, scores, labels, ids = 0., 0, [], [], []
    with torch.no_grad():
        for batch in loader:
            x,y,m = [batch[k].to(device) for k in ('image','target','valid')]
            out = model(x)
            if not torch.isfinite(out).all():
                raise RuntimeError('nonfinite evaluation logits')
            n = int(m.sum())
            total += masked_bce(out,y,m).item()*n
            count += n
            scores.append(out.cpu().numpy())
            labels.append(batch['raw_label'].numpy())
            ids.extend(batch['image_id'])
    scores, labels = np.concatenate(scores), np.concatenate(labels)
    ap = classification_map(labels,scores)
    valid, truth, pred = labels != 0, labels == 1, scores >= 0
    tp = int((pred & truth & valid).sum())
    fp = int((pred & ~truth & valid).sum())
    fn = int((~pred & truth & valid).sum())
    metrics = dict(loss=total/count, map=ap['map'], micro_f1=2*tp/max(2*tp+fp+fn,1),
                   positive_recall=tp/max(tp+fn,1), ap=dict(zip(CLASSES,ap['ap'].tolist())))
    return metrics, dict(scores=scores,labels=labels,image_ids=np.array(ids))


def subset_indices(ds, n, seed):
    """Cover all classes for interpretable tiny-set mAP, then seeded fill."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ds)).tolist()
    chosen, uncovered = [], set(range(20))
    labels = ds.raw_labels.numpy() == 1
    while uncovered:
        best = max((i for i in order if i not in chosen), key=lambda i: sum(labels[i,c] for c in uncovered))
        if not any(labels[best,c] for c in uncovered):
            raise ValueError('class coverage impossible')
        chosen.append(best)
        uncovered -= set(np.flatnonzero(labels[best]))
    if len(chosen)>n:
        raise ValueError('subset too small to cover all classes')
    chosen += [i for i in order if i not in chosen][:n-len(chosen)]
    return chosen


def write_report(run, cfg, result, history):
    path = ROOT/'report'/'archive'/'run_details'/f'{run.name}.md'
    path.parent.mkdir(parents=True,exist_ok=True)
    lines = [f'# 第四步训练报告：{run.name}', '', '## 目标与配置', '',
             f'- 模式：{cfg["mode"]}；随机种子 {cfg["seed"]}；设备 {result.get("device", "未启动")}。',
             f'- 配置：batch={cfg["batch_size"]}，旧层 LR={cfg["backbone_lr"]}，新层 LR={cfg["new_lr"]}，SGD momentum=0.9，weight_decay=0.0005。',
             '- 输入：整图缩放 224×224，ImageNet 归一化；256 维线性嵌入接 20 类线性分类头，无额外嵌入 ReLU。',
             '- 损失：对有效图像—类别条目求平均 BCE；忽略 difficult；此归约与论文按类求和存在缩放差异。',
             '- overfit：固定 32 张训练图，关闭翻转及 dropout，但所有层仍参与求导；baseline：完整 train，启用翻转及原有 dropout。',
             '- 官方 torchvision AlexNet ImageNet 权重；与原论文 Caffe 模型/预处理并非逐项等价。',
             f'- 原始记录：`runs/{run.name}/config.json`、`history.json`、`result.json`。', '', '## 指标', '',
             '| 阶段 | 评估集 | 训练 BCE | 评估 BCE | mAP (%) | micro-F1 (%) | 正类召回 (%) |', '|---|---|---:|---:|---:|---:|---:|']
    for row in history:
        m = row['metrics']
        train_loss = f'{row["train_bce"]:.6f}' if 'train_bce' in row else '—'
        lines.append(f'| {row["stage"]} | {row["split"]} | {train_loss} | {m["loss"]:.6f} | {m["map"]*100:.2f} | {m["micro_f1"]*100:.2f} | {m["positive_recall"]*100:.2f} |')
    lines += ['', f'- 状态：{result.get("status")}。', f'- 运行时间：{result.get("elapsed_seconds",0):.1f} 秒；训练峰值已分配显存：{result.get("peak_allocated_mib",0):.1f} MiB。',
              '- micro-F1 / 正类召回使用 sigmoid 0.5 阈值；AP 使用连续 logits。逐类 AP 位于 history.json/result.json。',
              '- baseline 训练 BCE 是该轮在线平均；overfit 是截至该步的在线累计平均。评估 BCE 均在 eval 模式重新计算。']
    if 'error' in result:
        lines += [f'- 错误：{result["error"]}']
    if 'overfit_passed' in result:
        lines += [f'- 预设拟合标准（BCE<0.05 且 micro-F1≥0.95）：{result["overfit_passed"]}。']
    lines += ['', '## 限制与改进方向', '',
              '- 小样本结果只证明可学习，不能说明泛化；短程 baseline 的 val 指标也不能与论文 test 指标直接比较。',
              '- 下一轮依据完整 val 的逐类 AP 与训练曲线决定训练时长和学习率；验证集只用于评估，未参与梯度更新。',
              '- 若拟合未通过，先检查 loss 缩放与学习率、正类召回和梯度；不要单凭负类占多数时 BCE 下降判断成功。',
              '- 完成整图基线后再实现多尺度 patch/SPP；当前不含 Fisher Layer/GMM/SVM。',
              '- 保留实验配置，后续对齐论文预处理、训练迭代数与学习率衰减，避免把调试配置当作论文配置。', '', '## 复现命令', '',
              '```powershell', cfg['command'], '```', '']
    path.write_text('\n'.join(lines),encoding='utf-8')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode',choices=['overfit','baseline'],default='overfit')
    p.add_argument('--steps',type=int,default=300)
    p.add_argument('--epochs',type=int,default=2)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--backbone-lr',type=float,default=.001)
    p.add_argument('--new-lr',type=float,default=.01)
    args = p.parse_args()
    if min(args.steps,args.epochs,args.batch_size)<=0 or args.workers<0:
        p.error('positive steps/epochs/batch-size and nonnegative workers required')
    cfg = vars(args).copy()
    cfg['command'] = 'python step04_train.py '+ ' '.join(f'--{k.replace("_","-")} {v}' for k,v in vars(args).items())
    cfg.update(torch=torch.__version__,torchvision=torchvision.__version__,preprocessing='resize224_whole_image_imagenet',
               embedding_activation='linear',loss_reduction='mean_valid_entries',momentum=.9,weight_decay=.0005)
    run = ROOT/'runs'/f'step04_{args.mode}_{datetime.now():%Y%m%d_%H%M%S}'
    run.mkdir(parents=True,exist_ok=False)
    history, result = [], {'status':'running'}
    start = time.perf_counter()
    try:
        torch.set_num_threads(2)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        if not torch.cuda.is_available():
            raise RuntimeError('this training experiment requires CUDA')
        device = 'cuda'
        result['device'] = torch.cuda.get_device_name(0)
        weights = ROOT/'weights/alexnet-owt-7be5be79.pth'
        sha = hashlib.sha256(weights.read_bytes()).hexdigest()
        if not sha.startswith('7be5be79'):
            raise ValueError('official weight hash mismatch')
        cfg['pretrained_sha256'] = sha
        (run/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
        root = ROOT/'datasets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012'
        train = VOCClassification(root,'train',baseline_transform(train=args.mode=='baseline'))
        val = VOCClassification(root,'val')
        if args.mode=='overfit':
            idx = subset_indices(train,32,args.seed)
            train_data = Subset(train,idx)
            eval_data = train_data
            split = 'train_subset32'
        else:
            idx = list(range(len(train)))
            train_data, eval_data, split = train, val, 'val_full5823'
        (run/'train_ids.json').write_text(json.dumps([train.ids[i] for i in idx]),encoding='utf-8')
        loader = DataLoader(train_data,batch_size=args.batch_size,shuffle=True,num_workers=args.workers,
                            persistent_workers=args.workers>0,pin_memory=True,generator=torch.Generator().manual_seed(args.seed))
        ev = DataLoader(eval_data,batch_size=args.batch_size,shuffle=False,num_workers=args.workers,
                        persistent_workers=args.workers>0,pin_memory=True)
        model = AlexNetBaseline(weights).to(device)
        optimizer = torch.optim.SGD(model.optimizer_groups(args.backbone_lr,args.new_lr),momentum=.9,weight_decay=.0005)
        torch.cuda.reset_peak_memory_stats()
        initial,_ = evaluate(model,ev,device)
        history.append(dict(stage='initial',split=split,metrics=initial))
        print('INITIAL',json.dumps(initial),flush=True)
        before = {k: p.detach().clone() for k,p in [('conv1',model.features[0].weight),('fc6',model.fc67[1].weight),('embedding',model.embedding.weight),('head',model.head.weight)]}
        best, best_stage = -1., None
        step, train_sum, train_count = 0,0.,0
        epochs = args.epochs if args.mode=='baseline' else (args.steps+len(loader)-1)//len(loader)
        for epoch in range(epochs):
            model.train(args.mode=='baseline')
            for batch in loader:
                x,y,m = [batch[k].to(device) for k in ('image','target','valid')]
                optimizer.zero_grad(set_to_none=True)
                loss = masked_bce(model(x),y,m)
                if not torch.isfinite(loss):
                    raise RuntimeError('nonfinite training loss')
                loss.backward()
                if step==0:
                    result['gradient_norms_first_step'] = {}
                    for name,module in [('backbone',model.features),('fc67',model.fc67),('embedding',model.embedding),('head',model.head)]:
                        grads = [p.grad for p in module.parameters()]
                        if any(g is None or not torch.isfinite(g).all() for g in grads):
                            raise RuntimeError('missing/nonfinite gradients: '+name)
                        result['gradient_norms_first_step'][name] = sum(float(g.norm()) for g in grads)
                optimizer.step()
                n = int(m.sum())
                train_sum += loss.item()*n
                train_count += n
                step += 1
                if step%25==0:
                    print(f'step={step} epoch={epoch+1} train_bce={train_sum/train_count:.6f}',flush=True)
                check = args.mode=='overfit' and (step%50==0 or step==args.steps)
                if check:
                    metrics, arrays = evaluate(model,ev,device)
                    history.append(dict(stage=f'step_{step}',split=split,metrics=metrics,train_bce=train_sum/train_count))
                    print('EVAL',step,json.dumps(metrics),flush=True)
                    if metrics['loss']<.05 and metrics['micro_f1']>=.95:
                        break
                if args.mode=='overfit' and step>=args.steps:
                    break
            if args.mode=='baseline':
                metrics,arrays = evaluate(model,ev,device)
                history.append(dict(stage=f'epoch_{epoch+1}',split=split,metrics=metrics,train_bce=train_sum/train_count))
                print('EVAL',epoch+1,json.dumps(metrics),flush=True)
                if metrics['map']>best:
                    best,best_stage = metrics['map'],epoch+1
                    torch.save({'model':model.state_dict(),'config':cfg,'classes':CLASSES,'epoch':epoch+1},run/'best.pt')
                train_sum,train_count = 0.,0
            (run/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
            if args.mode=='overfit' and (step>=args.steps or (step%50==0 and metrics['loss']<.05 and metrics['micro_f1']>=.95)):
                break
        result.update(final_metrics=metrics,steps=step,best_val_map=best if args.mode=='baseline' else None,best_epoch=best_stage)
        if args.mode=='overfit':
            result['overfit_passed'] = metrics['loss']<.05 and metrics['micro_f1']>=.95
        result['weight_changes_max_abs'] = {k:float((p.detach()-before[k]).abs().max()) for k,p in [('conv1',model.features[0].weight),('fc6',model.fc67[1].weight),('embedding',model.embedding.weight),('head',model.head.weight)]}
        if not all(v>0 for v in result['weight_changes_max_abs'].values()):
            raise RuntimeError('some parameter groups did not update')
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'config':cfg,'classes':CLASSES,'steps':step},run/'last.pt')
        np.savez_compressed(run/'final_predictions.npz',**arrays)
        # Check checkpoint restore using a fixed real-image batch.
        model.eval()
        probe = next(iter(ev))['image'][:2].to(device)
        with torch.no_grad(): expected = model(probe).cpu()
        checkpoint = torch.load(run/'last.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(checkpoint['model'])
        with torch.no_grad(): torch.testing.assert_close(model(probe).cpu(),expected)
        result['checkpoint_restore_passed'] = True
        result['status'] = 'completed'
    except Exception as exc:
        result.update(status='failed',error=repr(exc))
        raise
    finally:
        result['elapsed_seconds'] = time.perf_counter()-start
        result['peak_allocated_mib'] = torch.cuda.max_memory_allocated()/1024**2 if torch.cuda.is_available() else 0
        (run/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        (run/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        write_report(run,cfg,result,history)
        print('RUN',run,flush=True)
        print('RESULT',json.dumps(result),flush=True)


if __name__=='__main__':
    main()
