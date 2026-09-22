import unittest
import torch
from models.fisher_classifier import normalize_fisher
from data.voc_classification import masked_bce

class ClassifierTests(unittest.TestCase):
    def test_normalization(self):
        x=torch.tensor([[0.,0.,0.],[-4.,0.,9.]],dtype=torch.float64,requires_grad=True)
        y=normalize_fisher(x)
        torch.testing.assert_close(y[0],torch.zeros(3,dtype=x.dtype))
        torch.testing.assert_close(y[1].norm(),torch.tensor(1.,dtype=x.dtype))
        y.sum().backward();self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(torch.autograd.gradcheck(normalize_fisher,(torch.tensor([[-.3,.2,1.]],dtype=torch.float64,requires_grad=True),)))

    def test_masked_loss(self):
        x=torch.zeros(1,3,requires_grad=True)
        loss=masked_bce(x,torch.tensor([[1.,0.,1.]]),torch.tensor([[True,True,False]]))
        loss.backward();torch.testing.assert_close(x.grad,torch.tensor([[-.25,.25,0.]]))

if __name__=='__main__':unittest.main(verbosity=2)
