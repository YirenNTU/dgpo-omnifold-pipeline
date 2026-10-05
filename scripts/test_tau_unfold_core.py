import unittest
import numpy as np
from scripts.tau_unfold_core import indices, moment, response_edges, nested_sets, summarize_pseudoexperiments


class CoreTests(unittest.TestCase):
    def test_pseudo_bias_vs_centered_coverage(self):
        result=summarize_pseudoexperiments([9.,10.,11.],[1.,1.,1.],0.,10.)
        self.assertEqual(result['coverage68'],0.)
        self.assertEqual(result['centered_coverage68'],1.)
        self.assertEqual(result['pull_mean'],10.)
        self.assertEqual(result['pull_std'],1.)
        self.assertEqual(result['centered_pull_mean'],0.)
        self.assertEqual(result['std_over_mean_sigma'],1.)

    def test_pseudo_zero_uncertainty_rejected(self):
        with self.assertRaises(ValueError):summarize_pseudoexperiments([1,2],[0,1],1,1)

    def test_nested_response_sets(self):
        group=np.repeat([0,1],40); pool=np.r_[np.arange(20),np.arange(40,60)]
        a=nested_sets(group,pool,[.25,.5,1.],91)
        self.assertTrue(set(a[.25])<=set(a[.5])<=set(a[1.]))
        np.testing.assert_array_equal(a[1.],pool)
        for f,ids in a.items():
            self.assertEqual(len(ids),int(len(pool)*f))
            self.assertEqual(len(np.unique(ids)),len(ids))
            self.assertEqual(set(group[ids]),{0,1})
        again=nested_sets(group,pool,[.25,.5,1.],91)
        np.testing.assert_array_equal(a[.25],again[.25])

    def test_edges(self):
        np.testing.assert_array_equal(indices(np.array([-1,0,1]),10),[0,5,9])

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError): indices(np.array([np.nan]),10)

    def test_covariance(self):
        counts=np.array([30.,70.]); coeff=np.array([-1.,1.])
        value,error=moment(counts,np.diag(counts),coeff)
        self.assertAlmostEqual(value,.4)
        self.assertAlmostEqual(error,np.sqrt(.84/100))

    def test_offdiagonal_matters(self):
        counts=np.array([50.,50.]); coeff=np.array([-1.,1.])
        _,error=moment(counts,np.array([[50.,20.],[20.,50.]]),coeff)
        self.assertAlmostEqual(error,np.sqrt(.006))

    def test_mixed_kappa(self):
        # Each group's inverse kappa product belongs inside the moment.
        value,_=moment(np.array([20.,80.]),np.diag([20.,80.]),np.array([9.,-9.]))
        self.assertAlmostEqual(value,-5.4)

    def test_nonpositive_norm(self):
        with self.assertRaises(ValueError): moment(np.zeros(2),np.eye(2),np.ones(2))

    def test_nonuniform_edges(self):
        np.testing.assert_array_equal(indices(np.array([-1,-.5,0,1]),[-1,-.2,1]),[0,0,1,1])

    def test_sparse_edges_merged_for_both_models(self):
        x=np.linspace(-.79,.79,400)
        edges,report=response_edges({'truth':x,'pretrain':x*.9,'dgpo':x*.8},
                                   np.zeros(400,int),np.ones(400),10,20)
        self.assertLess(len(edges)-1,10)
        self.assertEqual(edges[0],-1); self.assertEqual(edges[-1],1)
        self.assertTrue(report['merges'])
        for row in report['final']: self.assertGreaterEqual(min(row['count']),20)

    def test_dense_bins_unchanged(self):
        x=np.linspace(-.999,.999,1000)
        edges,report=response_edges({'truth':x},np.zeros(1000,int),np.ones(1000),10,20)
        self.assertEqual(len(edges),11); self.assertEqual(report['merges'],[])

    def test_insufficient_group_not_dropped(self):
        x=np.linspace(-.9,.9,10)
        with self.assertRaisesRegex(ValueError,'Insufficient response support'):
            response_edges({'truth':x},np.zeros(10,int),np.ones(10),10,20)


if __name__=='__main__': unittest.main()
