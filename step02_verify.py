"""Run from the project folder: python step02_verify.py --require-cuda

Tests use unittest from the standard library; no new dependencies required.
"""
import argparse
import io
import json
from pathlib import Path
import runpy
import sys
import unittest

import numpy as np
import torch
from torch.func import functional_call
from models import FisherLayer

ROOT = Path(__file__).resolve().parent
encode = runpy.run_path(str(ROOT / 'step01_fisher'))['encode']
REPORT = {}


class FisherTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.layer = FisherLayer(3, 4).double()
        self.mu = torch.randn(3, 4, dtype=torch.double) * 0.3
        self.sigma = torch.rand(3, 4, dtype=torch.double) + 0.8
        self.layer.initialize_from_gmm(self.mu, self.sigma)
        self.x = torch.randn(2, 5, 4, dtype=torch.double)

    def test_numpy_and_mask(self):
        mask = torch.tensor([[True]*5, [True, False, True, False, True]])
        padded = self.x.clone()
        padded[~mask] = float('nan')
        padded.requires_grad_()
        actual = self.layer(padded, mask)
        expected = np.stack([encode(self.x[i, mask[i]].numpy(), self.mu.numpy(),
                                   self.sigma.numpy(), simplified=True)[0] for i in range(2)])
        np.testing.assert_allclose(actual.detach().numpy(), expected, atol=1e-12, rtol=1e-12)
        REPORT['numpy_max_absolute_error'] = float(np.max(np.abs(actual.detach().numpy()-expected)))
        actual.square().sum().backward()
        self.assertTrue(torch.isfinite(padded.grad).all())
        self.assertEqual(padded.grad[~mask].abs().sum().item(), 0)
        for p in self.layer.parameters():
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(p.grad.abs().sum().item(), 0)

    def test_gradcheck_x_w_b(self):
        layer = FisherLayer(2, 2).double()
        x = (torch.randn(1, 3, 2, dtype=torch.double)*0.2).requires_grad_()
        w = (torch.rand(2, 2, dtype=torch.double)+0.7).requires_grad_()
        b = (torch.randn(2, 2, dtype=torch.double)*0.2).requires_grad_()
        mask = torch.tensor([[True, False, True]])
        def f(x, w, b):
            return functional_call(layer, {'w': w, 'b': b}, (x, mask))
        self.assertTrue(torch.autograd.gradcheck(f, (x, w, b), eps=1e-6, atol=1e-5, rtol=1e-3))

    def test_invariance_and_analytic_case(self):
        fv = self.layer(self.x)
        torch.testing.assert_close(self.layer(self.x[:, [4, 0, 2, 1, 3]]), fv)
        torch.testing.assert_close(self.layer(self.x.repeat(1, 2, 1)), fv)
        layer = FisherLayer(1, 1).double().initialize_from_gmm([[0.]], [[1.]])
        result = layer(torch.tensor([[[-2.], [2.]], [[0.], [0.]]], dtype=torch.double))
        torch.testing.assert_close(result, torch.tensor([[0., 3/np.sqrt(2)], [0., -1/np.sqrt(2)]], dtype=torch.double))

    def test_validation(self):
        for mask in (torch.zeros(2, 5, dtype=torch.bool), torch.ones(2, 5), torch.ones(2, 4, dtype=torch.bool)):
            with self.assertRaises(ValueError):
                self.layer(self.x, mask)
        for x in (self.x[:, :0], self.x[..., :2], self.x.float(), self.x*float('nan')):
            with self.assertRaises(ValueError):
                self.layer(x)
        for sigma in (torch.zeros_like(self.sigma), -self.sigma, self.sigma*float('nan')):
            with self.assertRaises(ValueError):
                self.layer.initialize_from_gmm(self.mu, sigma)

    def test_optimizer_and_checkpoint(self):
        head = torch.nn.Linear(24, 2).double()
        optimizer = torch.optim.SGD(list(self.layer.parameters())+list(head.parameters()), lr=0.01)
        before = {n: p.detach().clone() for n, p in self.layer.named_parameters()}
        loss = torch.nn.functional.binary_cross_entropy_with_logits(head(self.layer(self.x)), torch.eye(2, dtype=torch.double))
        loss.backward()
        optimizer.step()
        for n, p in self.layer.named_parameters():
            self.assertFalse(torch.equal(p, before[n]))
        buffer = io.BytesIO()
        torch.save(self.layer.state_dict(), buffer)
        buffer.seek(0)
        restored = FisherLayer(3, 4).double()
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        torch.testing.assert_close(restored(self.x), self.layer(self.x))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_paper_dimensions(self):
        cpu = FisherLayer(32, 256)
        gpu = FisherLayer(32, 256).cuda()
        gpu.load_state_dict(cpu.state_dict())
        x = torch.randn(2, 100, 256)
        mask = torch.ones(2, 100, dtype=torch.bool)
        mask[1, 73:] = False
        xc = x.clone().requires_grad_()
        xg = x.cuda().requires_grad_()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        yc = cpu(xc, mask)
        yg = gpu(xg, mask.cuda())
        self.assertEqual(tuple(yg.shape), (2, 16384))
        torch.testing.assert_close(yg.cpu(), yc, atol=2e-5, rtol=2e-4)
        yc.square().mean().backward()
        yg.square().mean().backward()
        for a, b in [(xc.grad, xg.grad.cpu()), (cpu.w.grad, gpu.w.grad.cpu()), (cpu.b.grad, gpu.b.grad.cpu())]:
            self.assertTrue(torch.isfinite(b).all())
            self.assertGreater(b.abs().sum().item(), 0)
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-3)
        torch.cuda.synchronize()
        REPORT['gpu_peak_allocated_mib'] = torch.cuda.max_memory_allocated()/1024**2
        REPORT['gpu_output_shape'] = list(yg.shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-cuda', action='store_true')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if args.require_cuda and not torch.cuda.is_available():
        parser.error('CUDA is required but unavailable')
    REPORT.update(python=sys.version.split()[0], numpy=np.__version__, torch=torch.__version__,
                  cuda_build=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FisherTests))
    REPORT.update(tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                  skipped=len(result.skipped), passed=result.wasSuccessful())
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(REPORT, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps(REPORT, indent=2, ensure_ascii=False))
    sys.exit(0 if result.wasSuccessful() else 1)
