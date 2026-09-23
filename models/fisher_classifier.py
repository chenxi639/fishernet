"""Normalized Fisher representation and independent multi-label logits."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .dense_fisher_encoder import DenseFisherEncoder

def normalize_fisher(x,eps=1e-6,mode='global',num_components=32):
    if eps<=0:raise ValueError('positive epsilon required')
    # Continuous finite derivative at zero; approaches signed sqrt away from zero.
    power=x/torch.sqrt(x.abs()+eps)
    if mode=='intra':
        if x.ndim!=2 or x.shape[1]%(2*num_components):
            raise ValueError('intra normalization requires [B,2*K*D]')
        blocks=power.reshape(len(x),2,num_components,-1).permute(0,2,1,3)
        shape=blocks.shape
        blocks=torch.nn.functional.normalize(blocks.flatten(2),dim=-1,eps=eps).reshape(shape)
        power=blocks.permute(0,2,1,3).reshape_as(power)
    elif mode!='global':
        raise ValueError('normalization mode must be global or intra')
    return torch.nn.functional.normalize(power,p=2,dim=-1,eps=eps)

class FisherClassifier(nn.Module):
    def __init__(self,encoder):
        super().__init__();self.encoder=encoder
        ref=next(encoder.parameters())
        self.head=nn.Linear(2*encoder.fisher.num_components*256,20).to(ref)
        nn.init.normal_(self.head.weight,std=.01);nn.init.zeros_(self.head.bias)

    @classmethod
    def from_gmm(cls,bundle,gmm_run,device='cpu'):
        folder=Path(gmm_run)
        result=json.loads((folder/'result.json').read_text())
        config=json.loads((folder/'config.json').read_text())
        if result['status']!='passed' or not result['converged']:raise ValueError('verified converged GMM required')
        if hashlib.sha256((folder/'gmm.npz').read_bytes()).hexdigest()!=result['gmm_sha256']:
            raise ValueError('GMM checksum mismatch')
        encoder,transform,meta=DenseFisherEncoder.from_bundle(bundle,device)
        source=config['source_config']
        if meta['model_sha256']!=source['model_metadata']['model_sha256'] or source['longest_side']!=480:
            raise ValueError('GMM source and encoder mismatch')
        with np.load(folder/'gmm.npz',allow_pickle=False) as g:
            if not np.allclose(g['stds']**2,g['variances']):raise ValueError('invalid GMM stds')
            encoder.initialize_from_gmm(g['means'],g['stds'])
        return cls(encoder),transform,meta

    def forward(self,images):
        raw=self.encoder(images)['fisher']
        return self.head(normalize_fisher(raw))
