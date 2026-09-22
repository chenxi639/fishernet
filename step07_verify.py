"""Hand-check sampling/statistics helpers without images or network training."""
import unittest
import numpy as np
from step07_sample import select_indices,describe

class SamplingTests(unittest.TestCase):
    def test_reproducibility_and_cap(self):
        a=select_indices(100,16,42)
        np.testing.assert_array_equal(a,select_indices(100,16,42))
        self.assertEqual(len(np.unique(a)),16)
        self.assertTrue((a>=0).all() and (a<100).all())
        np.testing.assert_array_equal(select_indices(3,8,42),[0,1,2])
        with self.assertRaises(ValueError):select_indices(0,1,42)

    def test_statistics(self):
        s=describe(np.array([[0.,2.],[2.,2.]],dtype=np.float32))
        np.testing.assert_allclose(s['dimension_mean'],[1,2])
        np.testing.assert_allclose(s['dimension_variance'],[1,0])
        self.assertEqual(s['near_constant_dimensions'],1)
        self.assertAlmostEqual(s['covariance_effective_rank'],1)
        with self.assertRaises(ValueError):describe(np.array([[np.nan,1],[1,2]]))

if __name__=='__main__':unittest.main(verbosity=2)
