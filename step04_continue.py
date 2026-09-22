"""Continue whole-image baseline; no patch/SPP changes.

Resume model AND SGD momentum. Epoch-seeded data loading/dropout creates a
reproducible new continuation branch; old runs did not save their RNG states.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from data import VOCClassification, baseline_transform, masked_bce, CLASSES
from models.alexnet_baseline import AlexNetBaseline
from step04_train import evaluate, subset_indices

ROOT = Path(__file__).resolve().parent
WEAK = ('bottle', 'sofa', 'pottedplant')


def schedule(epoch, base):
    factor = .1 ** (int(epoch >= 7) + int(epoch >= 11))
    return [lr * factor for lr in base]


def plateau(history):
    """Predeclared practical plateau; not proof of statistical convergence."""
    recent = history[-4:]
    return (len(recent) == 4 and recent[-1]['epoch'] >= 12
            and np.ptp([r['val']['map'] for r in recent]) < .002
            and all(np.ptp([r['val']['ap'][c] for r in recent]) < .005 for c in WEAK))


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def report(run, cfg, result, history):
    lines = ['# 整图 AlexNet 基线续训报告', '', '## 目标与条件', '',
             '- 保持已有网络、224×224 预处理、masked BCE 与训练/验证划分；不加入多尺度 patch 或新池化模块。',
             f'- 起点：`{cfg["resume"]}`，恢复模型及 SGD 动量；旧运行未保存 RNG，不能宣称等价于一次未中断训练。',
             '- train=5717，完整 val=5823；另用固定512张训练图在 eval 模式诊断过拟合，ID已保存。',
             '- Windows / dinov3_pspnet / RTX4060 Laptop；环境详见 config.json。',
             '- batch=16；SGD momentum=0.9、weight_decay=0.0005；不重加权类别或改变采样。',
             '- 旧层/新层学习率：第3–6轮0.001/0.01，第7–10轮0.0001/0.001，第11轮起0.00001/0.0001。',
             '- 最多总计14轮；第12轮起，最近4轮整体mAP波动<0.2个百分点且三个弱类各自波动<0.5个百分点时提前停止。',
             '- 弱类在本次训练之前根据前次结果确定：bottle、sofa、pottedplant。最佳权重仍按整体val mAP选择，避免各类拼凑最优结果。', '',
             '## 每轮指标', '', '|轮次|在线训练BCE|训练探针BCE|val BCE|val mAP %|bottle AP %|sofa AP %|pottedplant AP %|',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in history:
        v = row['val']
        tr = f'{row["train_bce"]:.5f}' if row.get('train_bce') is not None else '—'
        lines.append(f'|{row["epoch"]}|{tr}|{row["probe"]["loss"]:.5f}|{v["loss"]:.5f}|{v["map"]*100:.2f}|'+
                     '|'.join(f'{v["ap"][c]*100:.2f}' for c in WEAK)+'|')
    lines += ['', '## 运行状态', '', f'- {result.get("status")}; 停止原因：{result.get("stop_reason", "运行中")}。',
              f'- 最佳轮次：{result.get("best_epoch")}; 最佳val mAP：{result.get("best_map",0)*100:.2f}%。',
              f'- 耗时 {result.get("elapsed_seconds",0):.1f} 秒；峰值已分配显存 {result.get("peak_allocated_mib",0):.1f} MiB。',
              f'- 源权重评估复核：{result.get("source_prediction_check")}; 优化器动量复核：{result.get("optimizer_check")}。',
              '', '## 限制与改进方向', '',
              '- train_probe只包含固定512张，不能当作完整训练集的无偏指标；用于同一组样本随时间的诊断。',
              '- 在线训练BCE含dropout/翻转，不能直接与eval模式的val BCE比较。',
              '- 最佳val指标存在选模偏差，不是独立test指标。到达轮数上限不代表收敛；平台判定只是单次运行的实用标准。',
              '- 若训练探针持续改善、val BCE变坏，应优先检查过拟合；若val AP仍升而BCE升，区分排序表现与置信度。',
              '- 弱类改善不足时下一步先分析错误样本和正负样本分布；本轮不改结构、不调阈值、不加类别权重。',
              f'- 原始配置/曲线/权重/预测：`runs/{run.name}/`。', '', '## 复现', '', '```powershell',cfg['command'],'```','']
    detail_dir=ROOT/'report'/'archive'/'run_details'
    detail_dir.mkdir(parents=True,exist_ok=True)
    (detail_dir/f'{run.name}.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--resume', type=Path, required=True)
    p.add_argument('--max-epoch', type=int, default=14)
    p.add_argument('--workers', type=int, default=2)
    args = p.parse_args()
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
    if 'optimizer' not in checkpoint or tuple(checkpoint['classes']) != CLASSES:
        raise ValueError('need checkpoint with matching classes and optimizer')
    source_cfg = checkpoint['config']
    if source_cfg.get('mode') != 'baseline':
        raise ValueError('only whole-image baseline checkpoints supported')
    start_epoch = checkpoint.get('epoch', source_cfg['epochs'])
    if args.max_epoch <= start_epoch or args.workers < 0:
        raise ValueError('max-epoch must exceed completed epoch; workers nonnegative')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for this run')
    run = ROOT/'runs'/f'step04_continue_{datetime.now():%Y%m%d_%H%M%S}'
    run.mkdir(parents=True)
    cfg = dict(source_cfg)
    cfg.update(resume=str(args.resume.resolve()), start_epoch=start_epoch, max_epoch=args.max_epoch,
               workers=args.workers, seed=42, milestones=[7,11], gamma=.1,
               source_sha256=hashlib.sha256(args.resume.read_bytes()).hexdigest(),
               torch=torch.__version__,gpu=torch.cuda.get_device_name(0),
               command=f'python step04_continue.py --resume "{args.resume}" --max-epoch {args.max_epoch} --workers {args.workers}')
    dump(run/'config.json', cfg)
    result, history = {'status':'running'}, []
    started = time.perf_counter()
    try:
        model = AlexNetBaseline(ROOT/'weights/alexnet-owt-7be5be79.pth').cuda()
        model.load_state_dict(checkpoint['model'])
        optimizer = torch.optim.SGD(model.optimizer_groups(),momentum=.9,weight_decay=.0005)
        optimizer.load_state_dict(checkpoint['optimizer'])
        # Check every restored momentum buffer before taking another step.
        loaded_state = optimizer.state_dict()['state']
        for key,state in checkpoint['optimizer']['state'].items():
            for name,value in state.items():
                if torch.is_tensor(value):
                    torch.testing.assert_close(loaded_state[key][name].cpu(), value, rtol=0, atol=0)
        result['optimizer_check'] = 'all saved momentum buffers match'
        del checkpoint
        root = ROOT/'datasets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012'
        train = VOCClassification(root,'train',baseline_transform(train=True))
        deterministic = VOCClassification(root,'train')
        val = VOCClassification(root,'val')
        probe_idx = subset_indices(deterministic,512,42)
        dump(run/'train_ids.json',train.ids)
        dump(run/'probe_ids.json',[train.ids[i] for i in probe_idx])
        dump(run/'val_ids.json',val.ids)
        ev = DataLoader(val,batch_size=16,num_workers=args.workers,persistent_workers=args.workers>0)
        probe = DataLoader(Subset(deterministic,probe_idx),batch_size=16,num_workers=args.workers,persistent_workers=args.workers>0)
        initial, arrays = evaluate(model,ev,'cuda')
        old_predictions = args.resume.parent/'final_predictions.npz'
        if old_predictions.exists() and args.resume.name=='last.pt':
            with np.load(old_predictions) as old:
                np.testing.assert_array_equal(arrays['image_ids'],old['image_ids'])
                np.testing.assert_array_equal(arrays['labels'],old['labels'])
                np.testing.assert_allclose(arrays['scores'],old['scores'],atol=2e-5,rtol=2e-4)
            result['source_prediction_check'] = 'full validation logits/IDs/labels match source'
        else:
            result['source_prediction_check'] = 'no matching saved predictions; evaluated loaded source'
        probe_metrics,_ = evaluate(model,probe,'cuda')
        history.append(dict(epoch=start_epoch,train_bce=None,val=initial,probe=probe_metrics))
        best, best_epoch = initial['map'], start_epoch
        best_metrics = initial
        global_steps = start_epoch*len(DataLoader(train,batch_size=16))
        base_lrs = [source_cfg['backbone_lr'],source_cfg['new_lr']]
        def save_checkpoint(path, epoch):
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),config=cfg,
                            classes=CLASSES,epoch=epoch,steps=global_steps),path)
        save_checkpoint(run/'best.pt',start_epoch)
        np.savez_compressed(run/'best_predictions.npz',**arrays)
        torch.cuda.reset_peak_memory_stats()
        print('SOURCE',json.dumps(initial),flush=True)
        for epoch in range(start_epoch+1,args.max_epoch+1):
            torch.manual_seed(42+epoch)
            np.random.seed(42+epoch)
            random.seed(42+epoch)
            lrs = schedule(epoch,base_lrs)
            for group,lr in zip(optimizer.param_groups,lrs): group['lr']=lr
            loader = DataLoader(train,batch_size=16,shuffle=True,num_workers=args.workers,pin_memory=True,
                                generator=torch.Generator().manual_seed(42+epoch))
            model.train()
            loss_sum, count = 0.,0
            for i,batch in enumerate(loader):
                x,y,m = [batch[k].cuda() for k in ('image','target','valid')]
                optimizer.zero_grad(set_to_none=True)
                loss = masked_bce(model(x),y,m)
                if not torch.isfinite(loss): raise RuntimeError('nonfinite loss')
                loss.backward()
                optimizer.step()
                n = int(m.sum())
                loss_sum += float(loss.detach())*n
                count += n
                global_steps += 1
                if (i+1)%100==0:
                    print(f'epoch={epoch} batch={i+1} train_bce={loss_sum/count:.5f}',flush=True)
            metrics,arrays = evaluate(model,ev,'cuda')
            probe_metrics,_ = evaluate(model,probe,'cuda')
            history.append(dict(epoch=epoch,train_bce=loss_sum/count,val=metrics,probe=probe_metrics,lrs=lrs))
            if metrics['map'] > best:
                best,best_epoch,best_metrics = metrics['map'],epoch,metrics
                save_checkpoint(run/'best.pt',epoch)
                np.savez_compressed(run/'best_predictions.npz',**arrays)
            save_checkpoint(run/'last.pt',epoch)
            np.savez_compressed(run/'final_predictions.npz',**arrays)
            result.update(best_epoch=best_epoch,best_map=best,best_metrics=best_metrics,last_epoch=epoch,
                          last_metrics=metrics,elapsed_seconds=time.perf_counter()-started,
                          peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2)
            dump(run/'history.json',history)
            dump(run/'result.json',result)
            report(run,cfg,result,history)
            print('EPOCH',epoch,json.dumps(dict(train_bce=loss_sum/count,val_bce=metrics['loss'],map=metrics['map'],
                                               weak={c:metrics['ap'][c] for c in WEAK},probe_loss=probe_metrics['loss'],lrs=lrs)),flush=True)
            if plateau(history):
                result['stop_reason']='predeclared practical plateau'
                break
        result.setdefault('stop_reason','max epoch reached; convergence not implied')
        # Load best and confirm deterministic predictions on real images.
        best_checkpoint = torch.load(run/'best.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(best_checkpoint['model'])
        with torch.no_grad():
            batch = next(iter(ev))
            predicted = model(batch['image'].cuda()).cpu().numpy()
        with np.load(run/'best_predictions.npz') as saved:
            np.testing.assert_allclose(predicted,saved['scores'][:len(predicted)],atol=2e-5,rtol=2e-4)
        result.update(status='completed',best_reload_check=True)
    except Exception as exc:
        result.update(status='failed',error=repr(exc))
        raise
    finally:
        result['elapsed_seconds']=time.perf_counter()-started
        dump(run/'history.json',history)
        dump(run/'result.json',result)
        report(run,cfg,result,history)
        print('RUN',run,flush=True)
        print('RESULT',json.dumps(result),flush=True)


if __name__=='__main__':
    main()
