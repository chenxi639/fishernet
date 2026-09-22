"""Infer independent class probabilities; not a mutually-exclusive classifier."""
import argparse
import json
from pathlib import Path
import torch
from PIL import Image
from models.baseline_bundle import load_baseline

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--bundle',type=Path,default=Path(__file__).resolve().parent/'artifacts/alexnet_baseline_v1')
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    args=p.parse_args()
    model,transform,meta=load_baseline(args.bundle,args.device)
    with Image.open(args.image) as im: x=transform(im.convert('RGB')).unsqueeze(0).to(args.device)
    with torch.no_grad():
        features=model.forward_features(x)
        probs=model.head(features).sigmoid()[0].cpu().tolist()
    print(json.dumps(dict(feature_shape=list(features.shape),probabilities=dict(sorted(
        zip(meta['classes'],probs),key=lambda item:-item[1]))),indent=2))
