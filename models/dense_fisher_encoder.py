"""Image-list -> dense SPP descriptors -> masked Fisher aggregation.

No optimizer or classification training here. Uninitialized Fisher output is
only allowed explicitly for interface checks, never a trained representation.
"""
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from data.patch_inputs import dense_regions,PatchTransform
from .spp_patch_encoder import SPPPatchEncoder
from .fisher_layer import FisherLayer


def pack_descriptors(flat,image_indices,batch_size):
    if flat.ndim!=2 or min(flat.shape)==0 or not flat.is_floating_point():
        raise ValueError('nonempty floating flat descriptors [R,D] required')
    if (image_indices.ndim!=1 or image_indices.shape[0]!=flat.shape[0]
            or image_indices.dtype!=torch.long or image_indices.device!=flat.device):
        raise ValueError('long image_indices [R] on descriptor device required')
    if type(batch_size) is not int or batch_size<=0 or ((image_indices<0)|(image_indices>=batch_size)).any():
        raise ValueError('invalid image index or batch_size')
    rows=[flat[image_indices==i] for i in range(batch_size)]
    if any(len(row)==0 for row in rows):raise ValueError('each image needs descriptors')
    counts=torch.tensor([len(row) for row in rows],device=flat.device)
    padded=pad_sequence(rows,batch_first=True,padding_value=0.)
    mask=torch.arange(padded.shape[1],device=flat.device)[None,:]<counts[:,None]
    return dict(descriptors=padded,mask=mask,counts=counts)


class DenseFisherEncoder(nn.Module):
    def __init__(self,patch_encoder,num_components=32,region_chunk_size=64):
        super().__init__()
        if type(region_chunk_size) is not int or region_chunk_size<=0:
            raise ValueError('region_chunk_size must be positive')
        self.patch_encoder=patch_encoder
        ref=next(patch_encoder.parameters())
        self.fisher=FisherLayer(num_components,256).to(device=ref.device,dtype=ref.dtype)
        self.register_buffer('gmm_initialized',torch.tensor(False,device=ref.device))
        self.region_chunk_size=region_chunk_size

    @classmethod
    def from_bundle(cls,bundle,device='cpu',longest_side=480,region_chunk_size=64):
        encoder,_,manifest=SPPPatchEncoder.from_bundle(bundle,device)
        model=cls(encoder,region_chunk_size=region_chunk_size).eval()
        metadata=dict(manifest,patch_longest_side=longest_side,patch_preprocessing='aspect_preserving',
                      patch_sizes=[64,96,128,160,192,224,256],patch_step=32)
        return model,PatchTransform(longest_side),metadata

    def initialize_from_gmm(self,means,stds):
        self.fisher.initialize_from_gmm(means,stds)
        self.gmm_initialized.fill_(True)
        return self

    def forward_patch_features(self,images):
        if not isinstance(images,(list,tuple)) or not images:raise ValueError('nonempty image list required')
        flat,indices,regions,details=[],[],[],[]
        ref=next(self.patch_encoder.parameters())
        for i,image in enumerate(images):
            if image.ndim!=3 or image.shape[0]!=3 or image.device!=ref.device or image.dtype!=ref.dtype:
                raise ValueError('each image must be [3,H,W] with model device/dtype')
            if not torch.isfinite(image).all():raise ValueError('image must be finite')
            roi=dense_regions(tuple(image.shape[-2:]),device=image.device,dtype=image.dtype)
            desc,info=self.patch_encoder(image[None],roi,True,region_chunk_size=self.region_chunk_size)
            flat.append(desc);indices.append(torch.full((len(desc),),i,device=desc.device,dtype=torch.long))
            regions.append(roi)
            projected=info['projected_regions'][:,1:]
            info['unique_projected_regions']=len(torch.unique(projected,dim=0))
            details.append(info)
        flat=torch.cat(flat);indices=torch.cat(indices)
        packed=pack_descriptors(flat,indices,len(images))
        return dict(packed,flat_descriptors=flat,image_indices=indices,regions=regions,details=details)

    def forward(self,images,allow_uninitialized=False):
        if not self.gmm_initialized.item() and not allow_uninitialized:
            raise RuntimeError('GMM is not initialized; use forward_patch_features or explicitly allow debug output')
        result=self.forward_patch_features(images)
        result['fisher']=self.fisher(result['descriptors'],result['mask'])
        result['gmm_initialized']=bool(self.gmm_initialized.item())
        return result
