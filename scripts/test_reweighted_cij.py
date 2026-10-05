import unittest
import numpy as np
from scripts.diagnose_reweighted_cij import analyze, features, prepare


class CijTests(unittest.TestCase):
    def data(self):
        rng=np.random.default_rng(91)
        a=rng.normal(size=(80,3)); a/=np.linalg.norm(a,axis=-1,keepdims=True)
        b=rng.normal(size=(80,3)); b/=np.linalg.norm(b,axis=-1,keepdims=True)
        return dict(event_id=np.arange(80),truth_a=a,truth_b=b,
                    sample_a=np.repeat(a[:,None],3,1),sample_b=np.repeat(b[:,None],3,1),
                    log_ratio=np.zeros((80,3)),event_weight=np.ones(80))

    def test_equal_weights_and_identical_truth(self):
        r=analyze(self.data(),[1,-1],'joint',30)
        np.testing.assert_allclose(r['results']['truth']['C'],r['results']['reweighted']['C'],atol=1e-14)
        self.assertAlmostEqual(r['comparison']['C_frobenius_error_change']['value'],0)
        self.assertAlmostEqual(r['weight_health']['event_ess_fraction'],1)

    def test_kappa_sign_and_axis_order(self):
        a=np.array([[1.,0,0]]); b=np.array([[0.,1,0]])
        f=features(a,b,[1,-1]).reshape(3,3)
        self.assertEqual(f[0,1],-9)
        self.assertEqual(np.count_nonzero(f),1)

    def test_log_offset_invariance(self):
        d=self.data(); d['log_ratio']=np.arange(240).reshape(80,3)/100
        for mode in ['joint','conditional']:
            n,w,_=prepare(d,[1,-1],mode)
            d2={**d,'log_ratio':d['log_ratio']+10000}
            n2,w2,_=prepare(d2,[1,-1],mode)
            np.testing.assert_allclose(n.sum(0)/w.sum(0)[:,None],n2.sum(0)/w2.sum(0)[:,None],atol=1e-12)

    def test_conditional_preserves_event_mass(self):
        d=self.data(); d['log_ratio']=np.arange(80)[:,None]+np.zeros((80,3))
        _,w,_=prepare(d,[1,-1],'conditional')
        np.testing.assert_allclose(w[:,2],1)

    def test_bad_ids_and_weights(self):
        d=self.data(); d['event_id'][1]=0
        with self.assertRaises(ValueError): prepare(d,[1,-1],'joint')
        d=self.data(); d['event_weight'][0]=-1
        with self.assertRaises(ValueError): prepare(d,[1,-1],'joint')

    def test_exact_zero_angles_retained(self):
        d=self.data(); d['truth_a'][:]=[1,0,0]; d['truth_b'][:]=[0,1,0]
        n,w,_=prepare(d,[1,-1],'joint')
        self.assertEqual((n.sum(0)/w.sum(0)[:,None])[0,1],-9)

    def test_analytic_spin_moment_and_channel_powers(self):
        # Exact spherical second-moment quadrature for
        # p(a,b|channel) ∝ 1 + kappaA*kappaB*C_xy*a_x*b_y.
        directions=np.concatenate((np.eye(3),-np.eye(3)))
        a=np.repeat(directions,6,axis=0);b=np.tile(directions,(6,1))
        a=np.tile(a,(2,1));b=np.tile(b,(2,1))
        kappas=np.repeat([[1.,-1.],[.5,-.4]],36,axis=0)
        ew=1+np.prod(kappas,axis=1)*.6*a[:,0]*b[:,1]
        d=dict(event_id=np.arange(72),truth_a=a,truth_b=b,sample_a=a[:,None],sample_b=b[:,None],
               log_ratio=np.zeros((72,1)),event_weight=ew)
        out=analyze(d,kappas,'joint',20)
        expected=np.zeros((3,3));expected[0,1]=.6
        np.testing.assert_allclose(out['results']['truth']['C'],expected,atol=1e-12)

    def test_known_density_ratio_closes(self):
        directions=np.concatenate((np.eye(3),-np.eye(3)))
        a=np.repeat(directions,6,axis=0);b=np.tile(directions,(6,1))
        ratio=1+.5*a[:,0]*b[:,1]
        sa=np.repeat(a,20,axis=0);sb=np.repeat(b,20,axis=0)
        ta=np.repeat(a,(20*ratio).astype(int),axis=0)
        tb=np.repeat(b,(20*ratio).astype(int),axis=0)
        d=dict(event_id=np.arange(len(sa)),truth_a=ta,truth_b=tb,sample_a=sa[:,None],sample_b=sb[:,None],
               log_ratio=np.log(np.repeat(ratio,20))[:,None],event_weight=np.ones(len(sa)))
        out=analyze(d,[1,1],'joint',20)
        self.assertAlmostEqual(out['results']['truth']['C'][0][1],.5)
        self.assertLess(out['results']['reweighted']['frobenius_error'],1e-12)
        self.assertAlmostEqual(out['comparison']['C_frobenius_error_change']['value'],-.5)

    def test_alternative_references_share_bootstrap_and_do_not_change_legacy(self):
        d=self.data()
        original=analyze(d,[1,-1],'joint',20)
        extended=analyze(d,[1,-1],'joint',20,references={'same':(d['truth_a'],d['truth_b'])})
        self.assertEqual(original['results'],extended['results'])
        self.assertEqual(original['comparison'],extended['comparison'])
        self.assertEqual(original['comparison'],extended['reference_comparisons']['same']['comparison'])
        np.testing.assert_array_equal(extended['reference_comparisons']['same']['reference_minus_stored_truth']['ci95'],0)


if __name__=='__main__': unittest.main()
