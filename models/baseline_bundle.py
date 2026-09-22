"""Load a baseline together with its exact preprocessing and class order."""
import hashlib
import json
from pathlib import Path
import torch
from data import baseline_transform, CLASSES
from .alexnet_baseline import AlexNetBaseline


def load_baseline(bundle, device='cpu', trainable=False):
    folder=Path(bundle)
    manifest=json.loads((folder/'baseline.json').read_text(encoding='utf-8'))
    if manifest['schema_version']!=1 or tuple(manifest['classes'])!=CLASSES:
        raise ValueError('unsupported bundle schema or class order')
    path=folder/manifest['model_file']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['model_sha256']:
        raise ValueError('model checksum mismatch')
    model=AlexNetBaseline()
    model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
    model.to(device).requires_grad_(trainable).train(trainable)
    transform=baseline_transform(False,manifest['image_size'])
    return model,transform,manifest
