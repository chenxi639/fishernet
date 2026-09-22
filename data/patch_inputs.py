"""Aspect-preserving inputs and deterministic resized-image patch grids."""
import torch
from torchvision.transforms import functional as TF
from .voc_classification import VOCClassification


class PatchTransform:
    def __init__(self, longest_side=480):
        if type(longest_side) is not int or longest_side<64:
            raise ValueError('longest_side must be an integer >=64')
        self.longest_side=longest_side

    def __call__(self,image):
        image=image.convert('RGB')
        w,h=image.size
        scale=self.longest_side/max(w,h)
        new_w,new_h=max(1,int(w*scale+.5)),max(1,int(h*scale+.5))
        if min(new_w,new_h)<64:
            raise ValueError('resized short side <64; choose a larger longest_side')
        resized=TF.resize(image,[new_h,new_w],antialias=True)
        tensor=TF.normalize(TF.to_tensor(resized),[.485,.456,.406],[.229,.224,.225])
        return dict(image=tensor,original_hw=(h,w),resized_hw=(new_h,new_w),
                    scale_xy=(new_w/w,new_h/h))


class VOCPatchDataset(VOCClassification):
    def __init__(self,root,split='train',longest_side=480):
        super().__init__(root,split,PatchTransform(longest_side))

    def __getitem__(self,index):
        sample=super().__getitem__(index)
        geometry=sample.pop('image')
        return dict(sample,**geometry)


def patch_collate(samples):
    """Keep images as a list: do not pad image pixels before convolution."""
    if not samples:raise ValueError('empty batch')
    return dict(images=[s['image'] for s in samples],image_ids=[s['image_id'] for s in samples],
                targets=torch.stack([s['target'] for s in samples]),
                valid_labels=torch.stack([s['valid'] for s in samples]),
                raw_labels=torch.stack([s['raw_label'] for s in samples]),
                geometry=[{k:s[k] for k in ('original_hw','resized_hw','scale_xy')} for s in samples])


def dense_regions(image_hw,sizes=(64,96,128,160,192,224,256),step=32,batch_index=0,
                  device=None,dtype=torch.float32):
    """Scale -> row -> column order, inclusive xyxy on RESIZED image.

    Only fully fitting squares; no appended off-grid edge region, no duplicates
    removed after feature projection. Borders can have <step uncovered pixels.
    This explicit policy is not yet an exact original-Caffe sampler match.
    """
    h,w=image_hw
    if any(type(v) is not int or v<=0 for v in (h,w,step)):
        raise ValueError('image dimensions and step must be positive integers')
    if type(batch_index) is not int or batch_index<0:
        raise ValueError('nonnegative integer batch_index required')
    sizes=tuple(sizes)
    if not sizes or any(type(s) is not int or s<=0 for s in sizes) or len(set(sizes))!=len(sizes):
        raise ValueError('unique positive integer sizes required')
    if not dtype.is_floating_point:raise ValueError('region dtype must be floating')
    boxes=[]
    for s in sizes:
        for y in range(0,h-s+1,step):
            for x in range(0,w-s+1,step):
                boxes.append([batch_index,x,y,x+s-1,y+s-1])
    if not boxes:raise ValueError('no patch fits; increase input size or change patch sizes')
    return torch.tensor(boxes,device=device,dtype=dtype)
