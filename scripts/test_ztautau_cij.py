import unittest
import tempfile
from types import SimpleNamespace
from pathlib import Path
import numpy as np
from scripts.diagnose_ztautau_cij import (angles,boost,tau_from_deltas,calibrate,align,check_core,load_core,run,
                                        reconstructed_targets,moment_diagnostics,reference_decomposition)
from scripts.export_ztautau_cij_candidates import match_context
from scripts.diagnose_reweighted_cij import features,analyze


class BridgeTests(unittest.TestCase):
    def sample(self):
        v=np.array([[5.,2.,1.,3.],[6.,-2.,3.,1.]])
        a=tau_from_deltas(v,np.zeros((2,2)))
        b=a.copy(); b[:,1:]*=-1
        return a,b,v,v*np.array([1,-1,-1,-1])

    def test_reference_core(self):
        repo=Path('/Users/yirenwu/Ztautau/TT2L-QC-Study')
        if not repo.is_dir(): self.skipTest('Optional external TT2L reference not installed')
        self.assertLess(check_core(load_core(repo),self.sample()),1e-9)

    def test_boost_roundtrip(self):
        a,_,_,_=self.sample(); b=np.array([[.1,.2,0],[-.2,0,.1]])
        np.testing.assert_allclose(boost(boost(a,b),-b),a,atol=1e-12)

    def physical_sample(self):
        rng=np.random.default_rng(732)
        n=24
        direction=rng.normal(size=(n,3));direction/=np.linalg.norm(direction,axis=-1,keepdims=True)
        energy=np.linspace(25,55,n)
        momentum=np.sqrt(energy**2-1.777**2)[:,None]*direction
        ta=np.column_stack((energy,momentum));tb=np.column_stack((energy,-momentum))
        rest_a=rng.normal(size=(n,3));rest_a*=.8/np.linalg.norm(rest_a,axis=-1,keepdims=True)
        rest_b=rng.normal(size=(n,3));rest_b*=.8/np.linalg.norm(rest_b,axis=-1,keepdims=True)
        tva=boost(np.column_stack((np.full(n,np.sqrt(.8**2+.14**2)),rest_a)),ta[:,1:]/ta[:,:1])
        tvb=boost(np.column_stack((np.full(n,np.sqrt(.8**2+.14**2)),rest_b)),tb[:,1:]/tb[:,:1])
        return ta,tb,tva,tvb

    def test_noncollinear_reference_and_lorentz_invariance(self):
        p=self.physical_sample()
        beta=np.broadcast_to([.12,-.09,.18],(len(p[0]),3))
        boosted=[boost(v,beta) for v in p]
        for before,after in zip(p,boosted):
            np.testing.assert_allclose(before[:,0]**2-(before[:,1:]**2).sum(1),
                                       after[:,0]**2-(after[:,1:]**2).sum(1),atol=2e-12)
        repo=Path('/Users/yirenwu/Ztautau/TT2L-QC-Study')
        if repo.is_dir():self.assertLess(check_core(load_core(repo),boosted),1e-9)
        # Nontrivial polarimeters: unit length, not merely ±k like the old fixture.
        for a in angles(*boosted):
            np.testing.assert_allclose(np.linalg.norm(a,axis=-1),1,atol=1e-12)
            self.assertGreater(np.std(a[:,1]),.1)

    def test_perfect_direction_closes_matched_but_not_full_truth(self):
        ta,tb,tva,tvb=self.physical_sample()
        # A controlled detector distortion, not a fitted correction.
        va=tva.copy();vb=tvb.copy();va[:,1]+=.03;vb[:,2]-=.02
        def theta(p):return np.arctan2(np.linalg.norm(p[:,1:3],axis=-1),p[:,3])
        def phi(p):return np.arctan2(p[:,2],p[:,1])
        d=np.stack([np.stack([theta(t)-theta(v),phi(t)-phi(v)],-1) for t,v in [(ta,va),(tb,vb)]],1)
        targets=reconstructed_targets(va,vb,ta,tb,d)
        truth=angles(ta,tb,tva,tvb)
        sa=tau_from_deltas(va,d[:,0]);sb=tau_from_deltas(vb,d[:,1])
        for mode,parents in [('raw',(sa,sb)),('calibrated',calibrate(sa,sb))]:
            aa,bb=angles(*parents,va,vb)
            data=dict(event_id=np.arange(len(d)),truth_a=truth[0],truth_b=truth[1],sample_a=aa[:,None],sample_b=bb[:,None],
                      log_ratio=np.zeros((len(d),1)),event_weight=np.ones(len(d)))
            refs=dict(recomputed_truth=truth,truth_tau_reco_visible=targets['truth_tau_reco_visible'],matched_target=targets[mode])
            report=analyze(data,[1,1],'joint',20,references=refs)
            self.assertGreater(report['results']['unweighted']['frobenius_error'],.01)
            matched=report['reference_comparisons']['matched_target']['results']
            for name in ('unweighted','reweighted'):
                self.assertLess(matched[name]['frobenius_error'],1e-12)
                np.testing.assert_allclose(matched[name]['C_minus_truth_ci95'],0,atol=1e-12)
            accounting=reference_decomposition(report)
            self.assertLess(accounting['closure']['reweighted']['max_accounting_error'],1e-12)

    def test_small_power_influence_not_hidden_by_weight_ess(self):
        n=100
        a=np.tile([1.,0,0],(n,1));b=np.tile([0.,1,0],(n,1))
        data=dict(event_id=np.arange(n),truth_a=a,truth_b=b,sample_a=a[:,None],sample_b=b[:,None],
                  log_ratio=np.zeros((n,1)),event_weight=np.ones(n))
        k=np.ones((n,2));k[0]=[1e-3,1e-3]
        diag=moment_diagnostics(data,k,'joint')
        self.assertAlmostEqual(diag['weight_health']['event_ess_fraction'],1)
        self.assertEqual(diag['inverse_abs_product_quantiles']['1'],1e6)
        self.assertGreater(diag['influence_top1pct_fraction']['reweighted'][0][1],.49)

    def test_local_reference_fixture(self):
        # Fixed values cross-checked with TT2L Core; no external checkout needed.
        a,b=angles(*self.sample())
        np.testing.assert_allclose(a,[[-1,0,0],[-1,0,0]],atol=1e-12)
        np.testing.assert_allclose(b,[[1,0,0],[1,0,0]],atol=1e-12)

    def test_float32_inputs_promoted_before_boost(self):
        p=[x.astype(np.float32) for x in self.sample()]
        a=angles(*p)
        b=angles(*[x.astype(np.float64) for x in p])
        for x,y in zip(a,b):
            self.assertEqual(x.dtype,np.float64)
            np.testing.assert_array_equal(x,y)
        self.assertEqual(tau_from_deltas(p[2],np.zeros((2,2),np.float32)).dtype,np.float64)

    def test_truth_difference_diagnostic(self):
        from scripts.diagnose_ztautau_cij import truth_comparison
        actual=list(angles(*self.sample()));stored=[v.copy() for v in actual]
        stored[1][0,2]=.002
        report=truth_comparison(actual,stored,['event0','event1'],1e-4)
        self.assertFalse(report['passed'])
        self.assertEqual(report['events_above_tolerance'],1)
        self.assertEqual(report['worst']['axis'],'n')

    def test_candidate_broadcast(self):
        a,b,v,w=self.sample()
        x,y=angles(a[:,None],b[:,None],v[:,None],w[:,None])
        np.testing.assert_allclose(np.linalg.norm(x,axis=-1),1)
        np.testing.assert_allclose(np.linalg.norm(y,axis=-1),1)

    def test_calibration(self):
        a,b,_,_=self.sample(); x,y=calibrate(a,b)
        np.testing.assert_allclose(x[:,1:]+y[:,1:],0)
        np.testing.assert_allclose(x[:,0]**2-(x[:,1:]**2).sum(1),1.777**2,atol=1e-10)

    def test_event_kappas(self):
        a=np.array([[1.,0,0],[1.,0,0]]); b=np.array([[0.,1,0],[0.,1,0]])
        k=np.array([[1,-1],[.5,-.5]])
        f=features(a[:,None],b[:,None],k)
        np.testing.assert_allclose(f[:,0,1],[-9,-36])

    def test_matching(self):
        x=np.arange(18).reshape(3,6)
        np.testing.assert_array_equal(match_context(x[[2,0]],x),[2,0])
        with self.assertRaises(ValueError): match_context(x[:1],np.concatenate((x,x[:1])))
        np.testing.assert_array_equal(align(np.array(['a','b']),np.array(['b','a'])),[1,0])

    def test_end_to_end_parquet_npz(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        repo=None  # Production end-to-end path must work without another repository.
        a,b,v,w=self.sample()
        ta,tb=angles(a,b,v,w)
        fields=dict(source_sample_index=[1,1],source_event_key=[10,20],event_weight=[1.,1.],analyzing_power_a=[1.,1.],analyzing_power_b=[-1.,-1.])
        for prefix,arr in zip(('lead_a_visible','lead_b_visible','truth_tau_a','truth_tau_b','truth_a_visible','truth_b_visible'),(v,w,a,b,v,w)):
            for i,c in enumerate(('E','px','py','pz')): fields[prefix+'_'+c]=arr[:,i]
        for leg,arr in [('A',ta),('B',tb)]:
            for i,c in enumerate(('k','r','n')): fields[f'truth_cos_theta_{leg}_{c}']=arr[:,i]
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            event_dir=root/'events';event_dir.mkdir()
            pq.write_table(pa.table(fields),event_dir/'part.parquet')
            (event_dir/'shape_metadata.json').write_text('{}')
            np.savez(root/'candidates.npz',source_sample_index=[1,1],source_event_key=[20,10],deltas=np.zeros((2,3,2,2)),truth_deltas=np.zeros((2,2,2)),log_ratio=np.zeros((2,3)))
            args=SimpleNamespace(events=event_dir,candidates=root/'candidates.npz',tt2l_repo=repo,output=root/'out',selection=None,kappa_signs=[1,1],weight_mode='conditional',bootstrap=20,seed=42,tolerance=1e-4)
            run(args)
            for mode in ('raw','calibrated'):
                self.assertTrue((root/'out'/f'{mode}.json').is_file())
                self.assertTrue((root/'out'/f'{mode}.png').is_file())
                self.assertTrue((root/'out'/f'{mode}_matched.png').is_file())
                self.assertTrue((root/'out'/f'{mode}_angular_moments.json').is_file())
            # Preserve a unit vector but deliberately differ from recomputation.
            fields['truth_cos_theta_A_k']=np.full(2,-np.cos(.002))
            fields['truth_cos_theta_A_r']=np.full(2,np.sin(.002))
            fields['truth_cos_theta_A_n']=np.zeros(2)
            pq.write_table(pa.table(fields),event_dir/'part.parquet')
            with self.assertRaisesRegex(ValueError,'Truth-angle mismatch'):
                run(args)
            args.allow_truth_mismatch=True
            run(args)
            import json
            report=json.loads((root/'out'/'raw.json').read_text())
            self.assertFalse(report['provenance']['truth_convention_passed'])
            self.assertTrue(report['provenance']['truth_mismatch_allowed'])
            self.assertEqual(report['provenance']['truth_source'],'stored parquet truth cosines')


if __name__=='__main__': unittest.main()
