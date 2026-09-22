"""Dense SPP/Fisher interface checks only: no optimizer, fitting or training."""
from datetime import datetime
import json
from pathlib import Path
import unittest
from PIL import Image
import torch
from data.patch_inputs import PatchTransform,VOCPatchDataset,patch_collate,dense_regions
from models.dense_fisher_encoder import DenseFisherEncoder,pack_descriptors

ROOT=Path(__file__).resolve().parent

class InterfaceTests(unittest.TestCase):
    def test_grid(self):
        for hw,n in [((320,480),490),((224,224),91),((64,64),1)]:
            r=dense_regions(hw)
            self.assertEqual(len(r),n)
            self.assertTrue((r[:,3]<hw[1]).all() and (r[:,4]<hw[0]).all())
            torch.testing.assert_close(r,dense_regions(hw))
        torch.testing.assert_close(dense_regions((65,97),sizes=(64,)),
                                   torch.tensor([[0.,0.,0.,63.,63.],[0.,32.,0.,95.,63.]]))
        for hw in [(32,480),(0,64)]:
            with self.assertRaises(ValueError):dense_regions(hw)

    def test_geometry(self):
        out=PatchTransform()(Image.new('L',(600,400)))
        self.assertEqual(out['image'].shape,(3,320,480))
        self.assertEqual(out['scale_xy'],(.8,.8))
        with self.assertRaises(ValueError):PatchTransform()(Image.new('RGB',(1000,10)))

    def test_packing_and_gradient(self):
        x=torch.arange(12,dtype=torch.float32).reshape(3,4).requires_grad_()
        p=pack_descriptors(x,torch.tensor([1,0,1]),2)
        self.assertEqual(p['counts'].tolist(),[1,2])
        torch.testing.assert_close(p['descriptors'][0,0],x[1])
        self.assertEqual(p['mask'].tolist(),[[True,False],[True,True]])
        p['descriptors'].sum().backward()
        torch.testing.assert_close(x.grad,torch.ones_like(x))
        with self.assertRaises(ValueError):pack_descriptors(x,torch.tensor([0,0,0]),2)

if __name__=='__main__':
    run=ROOT/'runs'/f'step06_dense_{datetime.now():%Y%m%d_%H%M%S}'
    run.mkdir(parents=True)
    result=dict(status='running',trained=False,gmm_fitted=False,optimizer_steps=0)
    try:
        torch.set_num_threads(2);torch.manual_seed(42)
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.matmul.allow_tf32=False
        tests=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(InterfaceTests))
        if not tests.wasSuccessful():raise RuntimeError('unit tests failed')
        device='cuda' if torch.cuda.is_available() else 'cpu'
        model,transform,metadata=DenseFisherEncoder.from_bundle(ROOT/'artifacts/alexnet_baseline_v1',device)
        ds=VOCPatchDataset(ROOT/'datasets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012')
        a=ds[0]
        b=next(ds[i] for i in range(1,50) if ds[i]['resized_hw']!=a['resized_hw'])
        batch=patch_collate([a,b])
        assert isinstance(batch['images'],list) and batch['targets'].shape==(2,20)
        torch.testing.assert_close(batch['valid_labels'],batch['raw_labels']!=0)
        torch.testing.assert_close(batch['targets'],(batch['raw_labels']==1).float())
        images=[im.to(device).requires_grad_() for im in batch['images']]
        versions={n:p._version for n,p in model.named_parameters()}
        with unittest.TestCase().assertRaises(RuntimeError):model(images)
        calls=[]
        hook=model.patch_encoder.conv.register_forward_hook(lambda m,i,o:calls.append(list(o.shape)))
        if device=='cuda':torch.cuda.reset_peak_memory_stats()
        out=model(images,allow_uninitialized=True)
        hook.remove()
        assert len(calls)==2 and out['fisher'].shape==(2,16384)
        assert torch.isfinite(out['fisher']).all()
        with torch.no_grad():
            separate=torch.cat([model([im],allow_uninitialized=True)['fisher'] for im in images])
            torch.testing.assert_close(out['fisher'],separate,rtol=2e-4,atol=2e-5)
            bad=out['descriptors'].detach().clone();bad[~out['mask']]=float('nan')
            torch.testing.assert_close(model.fisher(bad,out['mask']),out['fisher'])
            roi=out['regions'][0][:7]
            full=model.patch_encoder(images[0][None],roi)
            chunk=model.patch_encoder(images[0][None],roi,region_chunk_size=2)
            torch.testing.assert_close(full,chunk,rtol=2e-4,atol=2e-5)
        out['fisher'].square().mean().backward()
        norms={}
        for name,module in [('conv',model.patch_encoder.conv),('fc67',model.patch_encoder.fc67),
                            ('embedding',model.patch_encoder.embedding),('fisher',model.fisher)]:
            gs=[p.grad for p in module.parameters()]
            assert all(g is not None and torch.isfinite(g).all() for g in gs)
            norms[name]=sum(float(g.norm()) for g in gs)
            assert norms[name]>0
        assert all(im.grad is not None and torch.isfinite(im.grad).all() and im.grad.abs().sum()>0 for im in images)
        assert versions=={n:p._version for n,p in model.named_parameters()}
        result.update(status='passed',unit_tests=tests.testsRun,device=device,image_ids=batch['image_ids'],
                      geometry=batch['geometry'],patch_counts=out['counts'].tolist(),conv_shapes=calls,
                      unique_projected_counts=[d['unique_projected_regions'] for d in out['details']],
                      packed_shape=list(out['descriptors'].shape),fisher_shape=list(out['fisher'].shape),
                      batch_single_max_abs=float((out['fisher']-separate).abs().max().detach()),
                      chunk_max_abs=float((full-chunk).abs().max()),gradient_norms=norms,
                      parameters_unchanged=True,padding_nan_invariance=True,
                      descriptor_mean=float(out['flat_descriptors'].detach().mean()),
                      descriptor_std=float(out['flat_descriptors'].detach().std()),
                      max_allocated_mib=torch.cuda.max_memory_allocated()/1024**2 if device=='cuda' else None,
                      torch=torch.__version__,fisher_note='synthetic initialization, debug only; no classification metrics')
    except Exception as exc:
        result.update(status='failed',error=repr(exc));raise
    finally:
        (run/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print('RUN',run);print(json.dumps(result,indent=2))
