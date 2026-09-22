"""Sample frozen train-only patch descriptors; no GMM fitting or optimizer."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from data.patch_inputs import VOCPatchDataset
from data.voc_classification import CLASSES
from models.dense_fisher_encoder import DenseFisherEncoder

ROOT=Path(__file__).resolve().parent

def select_indices(total,cap,seed):
    if total<=0 or cap<=0:raise ValueError('total and cap must be positive')
    return np.sort(np.random.default_rng(seed).choice(total,min(total,cap),replace=False))

def describe(x):
    if x.ndim!=2 or len(x)<2 or not np.isfinite(x).all():
        raise ValueError('finite [N,D] descriptors, N>=2 required')
    z=x.astype(np.float64)
    mean=z.mean(0);var=z.var(0);norm=np.linalg.norm(z,axis=1)
    eig=np.maximum(np.linalg.eigvalsh(np.cov(z,rowvar=False)),0)
    return dict(shape=list(x.shape),dtype=str(x.dtype),finite=True,
                mean=float(z.mean()),std=float(z.std()),minimum=float(z.min()),maximum=float(z.max()),
                dimension_mean=mean.tolist(),dimension_variance=var.tolist(),
                variance_quantiles=np.quantile(var,[0,.25,.5,.75,1]).tolist(),
                near_constant_dimensions=int((var<1e-8).sum()),
                zero_norm_rows=int((norm<1e-8).sum()),
                l2_norm_quantiles=np.quantile(norm,[0,.01,.5,.99,1]).tolist(),
                covariance_eigenvalues=eig.tolist(),
                covariance_effective_rank=float(eig.sum()**2/(eig@eig)) if eig@eig>0 else 0.)

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images',type=int,default=256)
    p.add_argument('--per-image',type=int,default=128)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--bundle',type=Path,default=ROOT/'artifacts/alexnet_baseline_v1')
    args=p.parse_args()
    if args.images<1 or args.per_image<1 or args.seed<0:p.error('positive sample sizes and nonnegative seed required')
    run=ROOT/'runs'/f'step07_sample_{datetime.now():%Y%m%d_%H%M%S_%f}'
    run.mkdir(parents=True)
    result=dict(status='running',trained=False,gmm_fitted=False,optimizer_steps=0)
    write=lambda name,obj:(run/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    start=time.perf_counter()
    try:
        torch.set_num_threads(2);torch.manual_seed(args.seed)
        torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        model,_,metadata=DenseFisherEncoder.from_bundle(args.bundle,args.device)
        model.requires_grad_(False).eval()
        ds=VOCPatchDataset(ROOT/'datasets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012','train')
        if args.images>len(ds):raise ValueError('image count exceeds train split')
        chosen=select_indices(len(ds),args.images,args.seed)
        ids=[ds.ids[i] for i in chosen]
        val_ids=set((ds.root/'ImageSets/Main/val.txt').read_text().split())
        if set(ids)&val_ids:raise ValueError('train/val overlap detected')
        config=dict(images=args.images,per_image=args.per_image,seed=args.seed,device=args.device,
                    split='train',train_size=len(ds),bundle=str(args.bundle),model_metadata=metadata,
                    torch=torch.__version__,numpy=np.__version__,longest_side=480,
                    policy='uniform image subset; uniform without replacement within each image; cap per image',
                    normalization='raw 256-d embedding; no PCA/descriptor L2/standardization',
                    code_sha256={f:sha(ROOT/f) for f in ['step07_sample.py','data/patch_inputs.py',
                       'models/spp_patch_encoder.py','models/dense_fisher_encoder.py','models/alexnet_baseline.py']})
        write('config.json',config);write('image_ids.json',ids)
        versions={n:q._version for n,q in model.named_parameters()}
        values=[];roi_rows=[];grid_rows=[];owners=[];records=[];offsets=[0]
        if args.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            for j,index in enumerate(chosen):
                sample=ds[int(index)]
                out=model.forward_patch_features([sample['image'].to(args.device)])
                n=len(out['flat_descriptors'])
                # Image-ID-specific randomness keeps selections stable if subset size changes.
                key=int.from_bytes(hashlib.sha256(f'{args.seed}:{sample["image_id"]}'.encode()).digest()[:8],'little')
                selected=select_indices(n,args.per_image,key)
                take=torch.as_tensor(selected,device=args.device)
                x=out['flat_descriptors'][take].cpu().numpy().copy()
                if not np.isfinite(x).all():raise ValueError(f'nonfinite descriptors: {sample["image_id"]}')
                values.append(x);roi_rows.append(out['regions'][0][take,1:].cpu().numpy().copy())
                grid_rows.append(selected);owners.append(np.full(len(x),j,dtype=np.int32));offsets.append(offsets[-1]+len(x))
                records.append(dict(image_id=sample['image_id'],original_hw=sample['original_hw'],
                                    resized_hw=sample['resized_hw'],scale_xy=sample['scale_xy'],
                                    grid_count=n,sampled_count=len(x),
                                    unique_projected_count=out['details'][0]['unique_projected_regions']))
                if (j+1)%32==0:print(f'Sampled {j+1}/{len(chosen)} images, {offsets[-1]} descriptors',flush=True)
        extract_seconds=time.perf_counter()-start
        x=np.concatenate(values)
        stats=describe(x)
        path=run/'descriptors.npz'
        np.savez(path,descriptors=x,image_index=np.concatenate(owners),roi_xyxy=np.concatenate(roi_rows),
                 grid_index=np.concatenate(grid_rows),offsets=np.asarray(offsets,dtype=np.int64),image_ids=np.asarray(ids))
        with np.load(path,allow_pickle=False) as saved:
            assert np.array_equal(saved['descriptors'],x)
            assert np.array_equal(saved['image_ids'],ids)
            assert saved['offsets'][-1]==len(x)
            # Verify persisted ROI coordinates reproduce descriptors through the old SPP interface.
            count=min(8,offsets[1])
            roi=torch.tensor(np.column_stack([np.zeros(count),saved['roi_xyxy'][:count]]),dtype=torch.float32,device=args.device)
        with torch.inference_mode():
            check=model.patch_encoder(ds[int(chosen[0])]['image'][None].to(args.device),roi)
        np.testing.assert_allclose(check.cpu().numpy(),x[:count],rtol=2e-4,atol=2e-5)
        assert versions=={n:q._version for n,q in model.named_parameters()}
        assert not model.gmm_initialized.item()
        positives=(ds.raw_labels[chosen]==1).sum(0).tolist()
        write('images.json',records);write('statistics.json',stats)
        result.update(status='passed',descriptor_shape=list(x.shape),sampled_images=len(ids),
                      total_grid_regions=sum(r['grid_count'] for r in records),
                      projected_duplicates=sum(r['grid_count']-r['unique_projected_count'] for r in records),
                      class_positive_image_counts=dict(zip(CLASSES,positives)),
                      train_val_disjoint=True,parameters_unchanged=True,cache_reload_exact=True,
                      roi_replay_max_abs=float(np.abs(check.cpu().numpy()-x[:count]).max()),
                      extract_seconds=extract_seconds,total_seconds=time.perf_counter()-start,
                      cache_bytes=path.stat().st_size,cache_sha256=sha(path),
                      max_allocated_mib=torch.cuda.max_memory_allocated()/1024**2 if args.device.startswith('cuda') else None,
                      near_constant_dimensions=stats['near_constant_dimensions'],zero_norm_rows=stats['zero_norm_rows'])
    except Exception as exc:
        result.update(status='failed',error=repr(exc));raise
    finally:
        write('result.json',result)
        print('RUN',run);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
