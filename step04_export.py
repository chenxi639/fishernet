"""Export and verify a selected baseline; preserves original run checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader,Subset
from data import VOCClassification, CLASSES
from models.alexnet_baseline import AlexNetBaseline
from models.baseline_bundle import load_baseline

ROOT=Path(__file__).resolve().parent


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'artifacts/alexnet_baseline_v1')
    args=p.parse_args()
    torch.set_num_threads(2)
    source=args.run/'best.pt'
    if args.output.exists():
        # Recover only our own incomplete export, never replace a verified bundle.
        names={p.name for p in args.output.iterdir()}
        existing=args.output/'baseline.json'
        if (not names <= {'model.pt','baseline.json'} or not existing.exists()
                or json.loads(existing.read_text())['source_checkpoint']!=str(source.resolve())):
            raise ValueError('output exists; use a new bundle directory')
    else:
        args.output.mkdir(parents=True)
    # CPU/GPU equality checks require consistent arithmetic: CUDA convolution
    # defaults may use TF32. This setting only applies to this verification run.
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    checkpoint=torch.load(source,map_location='cpu',weights_only=False)
    cfg=checkpoint['config']
    result=json.loads((args.run/'result.json').read_text(encoding='utf-8'))
    if result['status']!='completed': raise ValueError('cannot export incomplete run')
    model_file=args.output/'model.pt'
    torch.save(checkpoint['model'],model_file)
    manifest=dict(schema_version=1,architecture='AlexNet + linear4096to256 + linear256to20',
                  model_file='model.pt',model_sha256=hashlib.sha256(model_file.read_bytes()).hexdigest(),
                  image_size=cfg.get('image_size',224),classes=list(CLASSES),
                  preprocessing='RGB; resize whole image square; ImageNet mean/std; no eval augmentation',
                  feature_dim=256,output_dim=20,source_checkpoint=str(source.resolve()),
                  source_checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  validation_split='VOC2012 val (5823 images), used for model selection',
                  validation_metrics=result.get('best_metrics',result.get('final_metrics')),
                  best_epoch=result.get('best_epoch'),limitations='whole-image local baseline; not full FisherNet or independent test result')
    (args.output/'baseline.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model,transform,metadata=load_baseline(args.output,device)
    source_model=AlexNetBaseline().to(device)
    source_model.load_state_dict(checkpoint['model']); source_model.eval()
    ds=VOCClassification(ROOT/'datasets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012','val',transform)
    batch=next(iter(DataLoader(Subset(ds,[0,100,500,1000]),batch_size=4)))
    with torch.no_grad():
        x=batch['image'].to(device)
        expected=source_model(x)
        actual=model(x)
        feats=model.forward_features(x)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert actual.shape==(4,20) and feats.shape==(4,256)
    assert torch.isfinite(feats).all() and torch.isfinite(actual).all()
    assert not any(p.requires_grad for p in model.parameters()) and not model.training
    # CPU bundle also loads and agrees with GPU within float32 tolerance.
    cpu_model,_,_=load_baseline(args.output,'cpu')
    with torch.no_grad(): cpu=cpu_model(batch['image'])
    torch.testing.assert_close(cpu,actual.cpu(),rtol=2e-4,atol=2e-5)
    verification=dict(passed=True,device=device,image_ids=batch['image_id'],feature_shape=list(feats.shape),
                      logit_shape=list(actual.shape),export_max_error=float((actual-expected).abs().max()),cpu_gpu_match=True,
                      comparison_precision='strict FP32; TF32 disabled for export check',
                      cpu_gpu_max_abs=float((cpu-actual.cpu()).abs().max()))
    (args.output/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    print(json.dumps(dict(bundle=str(args.output),**verification),indent=2))
