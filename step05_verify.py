"""Minimal SPP validation; no training, dense patch sampler, or GMM fitting."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import unittest
import torch
from torchvision.ops import roi_pool
from data import VOCClassification
from models.spp_patch_encoder import SPPPatchEncoder,project_regions

ROOT=Path(__file__).resolve().parent


class PoolTests(unittest.TestCase):
    def test_hand_pool_and_gradient(self):
        x=torch.arange(16,dtype=torch.float32).reshape(1,1,4,4).requires_grad_()
        roi=torch.tensor([[0.,0.,0.,3.,3.]])
        y=roi_pool(x,roi,(2,2),1.)
        torch.testing.assert_close(y,torch.tensor([[[[5.,7.],[13.,15.]]]]))
        y.sum().backward()
        expected=torch.zeros_like(x).flatten();expected[[5,7,13,15]]=1
        torch.testing.assert_close(x.grad.flatten(),expected)

    def test_coordinates_and_errors(self):
        roi=torch.tensor([[0.,0.,0.,223.,223.],[1.,16.,32.,63.,95.]])
        out=project_regions(roi,(224,224),(13,13),2)
        torch.testing.assert_close(out,torch.tensor([[0.,0.,0.,12.,12.],[1.,1.,2.,4.,6.]]))
        for bad in (torch.tensor([[2.,0.,0.,10.,10.]]),torch.tensor([[0.,10.,0.,1.,2.]]),
                    torch.tensor([[0.,-1.,0.,10.,10.]]),torch.tensor([[0.,0.,0.,224.,10.]])):
            with self.assertRaises(ValueError):project_regions(bad,(224,224),(13,13),2)

    def test_batch_selection_and_small_roi(self):
        x=torch.stack([torch.full((1,4,4),2.),torch.full((1,4,4),9.)])
        y=roi_pool(x,torch.tensor([[1.,1.,1.,1.,1.],[0.,0.,0.,3.,3.]]),(6,6),1.)
        torch.testing.assert_close(y[0],torch.full((1,6,6),9.))
        torch.testing.assert_close(y[1],torch.full((1,6,6),2.))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--bundle',type=Path,default=ROOT/'artifacts/alexnet_baseline_v1')
    args=p.parse_args()
    run=ROOT/'runs'/f'step05_spp_{datetime.now():%Y%m%d_%H%M%S}'
    run.mkdir(parents=True)
    result={'status':'running','trained':False}
    try:
        torch.set_num_threads(2);torch.manual_seed(42)
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.matmul.allow_tf32=False
        test=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PoolTests))
        if not test.wasSuccessful():raise RuntimeError('pool unit tests failed')
        device='cuda' if torch.cuda.is_available() else 'cpu'
        encoder,transform,manifest=SPPPatchEncoder.from_bundle(args.bundle,device)
        encoder.eval()
        ds=VOCClassification(ROOT/'datasets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012','train',transform)
        items=[ds[0],ds[100]]
        images=torch.stack([x['image'] for x in items]).to(device)
        h,w=images.shape[-2:]
        regions=torch.tensor([[0,0,0,w-1,h-1],[0,0,0,w//2-1,h//2-1],
                              [0,w//4,h//4,3*w//4-1,3*h//4-1],
                              [1,0,0,w-1,h-1],[1,w//2,h//2,w-1,h-1]],dtype=images.dtype,device=device)
        images.requires_grad_()
        if device=='cuda':torch.cuda.reset_peak_memory_stats()
        descriptors,details=encoder(images,regions,True)
        assert descriptors.shape==(5,256) and torch.isfinite(descriptors).all()
        with torch.no_grad():
            # Equal region independently extracted must agree; ordering/batching
            # may not change descriptor identity in eval mode.
            single=encoder(images[:1],regions[:3])
            torch.testing.assert_close(single,descriptors[:3],rtol=2e-4,atol=2e-5)
            perm=torch.tensor([4,2,0,3,1],device=device)
            reordered=encoder(images,regions[perm])
            torch.testing.assert_close(reordered,descriptors[perm],rtol=2e-4,atol=2e-5)
        descriptors.square().mean().backward()
        grads={}
        for name,module in [('conv',encoder.conv),('fc67',encoder.fc67),('embedding',encoder.embedding)]:
            group=[p.grad for p in module.parameters()]
            assert all(g is not None and torch.isfinite(g).all() for g in group)
            grads[name]=sum(float(g.norm()) for g in group)
            assert grads[name]>0
        assert torch.isfinite(images.grad).all() and images.grad.abs().sum()>0
        if device=='cuda':
            cpu_encoder,_,_=SPPPatchEncoder.from_bundle(args.bundle,'cpu');cpu_encoder.eval()
            with torch.no_grad():cpu=cpu_encoder(images.detach().cpu(),regions.cpu())
            torch.testing.assert_close(cpu,descriptors.detach().cpu(),rtol=3e-4,atol=3e-5)
            result['cpu_gpu_max_abs']=float((cpu-descriptors.detach().cpu()).abs().max())
        result.update(status='passed',unit_tests=test.testsRun,device=device,bundle=str(args.bundle),
                      image_ids=[x['image_id'] for x in items],image_shape=list(images.shape),
                      descriptor_shape=list(descriptors.shape),feature_shape=details['feature_shape'],
                      pooled_shape=details['pooled_shape'],regions=regions.cpu().tolist(),
                      projected_regions=details['projected_regions'].cpu().tolist(),gradient_norms=grads,
                      max_allocated_mib=torch.cuda.max_memory_allocated()/1024**2 if device=='cuda' else None,
                      torch=torch.__version__,coordinate_convention='inclusive resized pixels, stride16 round-half-up/clamp')
    except Exception as exc:
        result.update(status='failed',error=repr(exc));raise
    finally:
        (run/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print('RUN',run)
        print(json.dumps(result,indent=2))
