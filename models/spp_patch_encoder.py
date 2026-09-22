"""Initial AlexNet region-max-pooling implementation of FisherNet section 3.2.

Each region produces a single 6x6 grid, not a concatenated 1/2/4 pyramid.
Coordinates: [batch_index,x1,y1,x2,y2], inclusive pixels of the resized input.
Project by stride16 with round-half-up then clamp to conv5 bounds. This explicit
engineering convention has not been matched against original Caffe code.
"""
import torch
from torch import nn
from torchvision.ops import roi_pool
from .baseline_bundle import load_baseline


def project_regions(regions, image_hw, feature_hw, batch_size):
    if regions.ndim!=2 or regions.shape[1]!=5 or regions.shape[0]==0:
        raise ValueError('regions must be nonempty [R,5]')
    if not regions.is_floating_point() or not torch.isfinite(regions).all() or regions.requires_grad:
        raise ValueError('finite floating regions without coordinate gradients required')
    index=regions[:,0]
    if ((index<0)|(index>=batch_size)|(index!=index.floor())).any():
        raise ValueError('invalid region batch index')
    h,w=image_hw
    x1,y1,x2,y2=regions[:,1:].unbind(1)
    if ((x1<0)|(y1<0)|(x2>=w)|(y2>=h)|(x2<x1)|(y2<y1)).any():
        raise ValueError('inclusive region coordinates must be inside resized image')
    out=regions.clone()
    xy=torch.floor(regions[:,1:]/16+0.5)
    xy[:,[0,2]]=xy[:,[0,2]].clamp(0,feature_hw[1]-1)
    xy[:,[1,3]]=xy[:,[1,3]].clamp(0,feature_hw[0]-1)
    out[:,1:]=xy
    return out


class SPPPatchEncoder(nn.Module):
    def __init__(self, baseline):
        super().__init__()
        if not isinstance(baseline.features[-1],nn.MaxPool2d):
            raise ValueError('expected AlexNet final max pool')
        self.conv=nn.Sequential(*list(baseline.features.children())[:-1])
        self.fc67=baseline.fc67
        self.embedding=baseline.embedding

    @classmethod
    def from_bundle(cls,bundle,device='cpu'):
        baseline,transform,manifest=load_baseline(bundle,device,trainable=True)
        encoder=cls(baseline).to(device)
        return encoder,transform,manifest

    def forward(self, images, regions, return_details=False, region_chunk_size=None):
        if images.ndim!=4 or images.shape[1]!=3 or min(images.shape)<1:
            raise ValueError('images must be nonempty [B,3,H,W]')
        if regions.device!=images.device or regions.dtype!=images.dtype:
            raise ValueError('images/regions must share dtype and device')
        if region_chunk_size is not None and (type(region_chunk_size) is not int or region_chunk_size<=0):
            raise ValueError('region_chunk_size must be positive or None')
        ref=next(self.parameters())
        if images.device!=ref.device or images.dtype!=ref.dtype:
            raise ValueError('images and encoder must share dtype and device')
        feature_map=self.conv(images)
        projected=project_regions(regions,images.shape[-2:],feature_map.shape[-2:],images.shape[0])
        chunk_size=region_chunk_size or len(projected)
        pieces=[]
        for chunk in projected.split(chunk_size):
            pooled=roi_pool(feature_map,chunk,output_size=(6,6),spatial_scale=1.)
            pieces.append(self.embedding(self.fc67(pooled.flatten(1))))
        descriptors=torch.cat(pieces)
        if return_details:
            return descriptors,{'feature_shape':list(feature_map.shape),'pooled_shape':[len(projected),feature_map.shape[1],6,6],
                                'projected_regions':projected.detach()}
        return descriptors
