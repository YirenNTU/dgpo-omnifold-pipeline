import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

import numpy as np
import torch
import yaml

from scripts.audit_conditional_tau_ratio import (audit_split, class_weights,
    weighted_bce, metrics, attribution, new_model, prepare, endpoint, merge_scores, compare_audits, worker, ARMS)


class WeightedAuditTest(unittest.TestCase):
    def test_identity_split_is_paired_order_independent(self):
        ids=np.arange(5000).astype(str)
        split=audit_split(ids,42)
        np.testing.assert_array_equal(audit_split(ids[::-1],42),split[::-1])
        self.assertEqual(set(split),{0,1,2})
        self.assertTrue(.57 < np.mean(split==0) < .63)
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            audit_split(['1','1'],42)

    def test_class_normalization_global_not_minibatch(self):
        base=np.array([1.,2.,3.,0.])
        logits=np.array([1000.,1001.,998.,3000.])
        wp,wq=class_weights(base,logits,True)
        self.assertAlmostEqual(wp.mean(),1.)
        self.assertAlmostEqual(wq.mean(),1.)
        self.assertEqual(wq[-1],0.)
        self.assertNotAlmostEqual(wq[:2].mean(),1.)
        p,q=torch.tensor([.2,-1.,2.,3.]),torch.tensor([1.,.4,-.3,2.])
        wp,wq=torch.tensor(wp),torch.tensor(wq)
        whole=weighted_bce(p,q,wp,wq)
        halves=(weighted_bce(p[:2],q[:2],wp[:2],wq[:2])+
                weighted_bce(p[2:],q[2:],wp[2:],wq[2:]))/2
        torch.testing.assert_close(whole,halves)

    def test_condition_only_control_equal_weights_chance(self):
        p=np.linspace(-2,2,100)
        m=metrics(p,p,np.ones(100),np.zeros(100),True)
        self.assertAlmostEqual(m['auc'],.5)
        m=metrics(p,p,np.ones(100),p*5,True)
        self.assertGreater(m['auc_gap'],.2)

    def test_known_exact_ratio_closes_weighted_population(self):
        truth=np.r_[np.ones(75),np.zeros(25)]
        generated=np.r_[np.ones(25),np.zeros(75)]
        ratio=np.where(generated==1,3.,1/3)
        raw=metrics(truth,generated,np.ones(100),np.log(ratio),False)
        weighted=metrics(truth,generated,np.ones(100),np.log(ratio),True)
        self.assertAlmostEqual(raw['auc'],.75)
        self.assertAlmostEqual(weighted['auc'],.5)
        flat=metrics(truth*0,generated*0,np.ones(100),np.log(ratio),True)
        self.assertAlmostEqual(flat['bce'],np.log(2))

    def test_matched_endpoint_comparison(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for arm in ARMS[:2]:
                (root/arm).mkdir()
                np.savez(root/arm/'audit_test_scores.npz',source_ids=np.arange(50),
                    source_log_ratio=np.zeros(50),event_weight=np.ones(50),
                    truth_logits=np.ones(50),generated_logits=np.zeros(50))
            comparison=compare_audits(root,20,42)
            self.assertEqual(comparison['weighted_minus_unweighted'],dict(auc_gap=0.,bce=0.))
            self.assertEqual(comparison['ci95']['auc_gap'],[0.,0.])

    def test_error_and_error_change_accounting_exact(self):
        rng=np.random.default_rng(12)
        t,g=rng.normal(size=(60,9)),rng.normal(size=(60,9))
        base=np.ones(60);q=rng.normal(size=60)
        report=attribution(t,g,base,q,np.arange(60),np.repeat([11,22],30))
        self.assertLess(report['reconstruction_error'],1e-12)
        self.assertAlmostEqual(report['projection_sum'],report['error_norm'])
        self.assertAlmostEqual(report['change_contribution_sum'],report['error_change'])
        row=report['top_error_events'][0];i=int(row['source_id'])
        take=np.arange(60)!=i
        w,_=class_weights(base[take],q[take],True)
        _,rw=class_weights(base[take],q[take],True)
        error=np.linalg.norm(np.average(g[take],weights=rw,axis=0)-np.average(t[take],weights=w,axis=0))
        self.assertAlmostEqual(row['leave_one_pair_out_error_change'],error-report['error_norm'])

    def test_joint_arms_identical_initialization_and_finite_gradient(self):
        a=dict(condition=np.ones((10,3)),candidate_truth=np.ones((10,21)))
        cfg=dict(seed=42,hidden=16,dropout=0.,relative_dim=6)
        left,right=[new_model(cfg,a,arm) for arm in ARMS[:2]]
        for k,v in left.state_dict().items():
            torch.testing.assert_close(v,right.state_dict()[k],rtol=0,atol=0)
        c,t,g=torch.randn(10,3),torch.randn(10,21),torch.randn(10,21)
        loss=weighted_bce(left(c,t),left(c,g),torch.ones(10),torch.ones(10))
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in left.parameters()))
        m=new_model(cfg,a,'weighted_condition').eval()
        torch.testing.assert_close(m(c,t),m(c,g))

    def test_prepare_and_endpoint_only_external_events(self):
        # End-to-end preparation + fresh checkpoint evaluation, without Ray/GPU.
        rng=np.random.default_rng(4);n=500
        dirs=rng.normal(size=(n,3));dirs/=np.linalg.norm(dirs,axis=1)[:,None]
        arrays=dict(split=np.r_[np.zeros(100),np.full(400,2)],
            source_ids=np.arange(n).astype(str),event_weight=np.ones(n),
            condition=rng.normal(size=(n,3)).astype('float32'),
            candidate_truth=rng.normal(size=(n,21)).astype('float32'),
            candidate_generated=rng.normal(size=(n,21)).astype('float32'),
            truth_a=dirs,truth_b=dirs,sample_a=dirs,sample_b=dirs,
            kappas=np.ones((n,2)),category=np.full(n,11),
            tau_truth=rng.normal(size=(n,15)),tau_generated=rng.normal(size=(n,15)))
        cfg=dict(seed=42,hidden=16,dropout=0.,relative_dim=6,min_steps=1000,bootstrap=20,workers=1)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'manifest.json').write_text(json.dumps(dict(representation='tau',candidate_count=1,
                relative_dim=6,backbone_cache='frozen')))
            np.savez(root/'prepared.npz',**arrays)
            q=rng.normal(size=400)
            np.savez(root/'test_scores.npz',source_ids=arrays['source_ids'][100:],
                log_ratio=q,generated_logits=q,truth_logits=q*0,saved_h4_log_ratio=q*0)
            a,tail,meta=prepare(root,cfg)
            self.assertEqual(len(a['source_ids']),400)
            self.assertFalse(set(a['source_ids']).intersection(arrays['source_ids'][:100]))
            self.assertEqual(set(tail['error_attribution']),{'tau','angular','cij'})
            cfg['checkpoint']=str(root/'best.pt')
            m=new_model(cfg,a,'weighted_joint')
            torch.save(dict(state_dict=m.state_dict(),epoch=1,steps=5,val_bce=.7),cfg['checkpoint'])
            test=a['split']==2
            np.savez(root/'test-rank-00.npz',positions=np.arange(test.sum()),
                source_ids=a['source_ids'][test],truth_logits=np.ones(test.sum()),
                generated_logits=np.zeros(test.sum()))
            (root/'fit_status.json').write_text(json.dumps(dict(steps=5)))
            out=endpoint(cfg,a,'weighted_joint')
            self.assertFalse(out['minimum_fit_budget_met'])
            self.assertEqual(len(out['auc_ci95']),2)
            self.assertTrue((root/'audit_test_scores.npz').is_file())

    def test_distributed_score_merging(self):
        ids=np.arange(73).astype(str)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for rank in range(16):
                idx=np.arange(rank,len(ids),16)
                np.savez(root/f'test-rank-{rank:02d}.npz',positions=idx,source_ids=ids[idx],
                    truth_logits=idx,generated_logits=-idx)
            p,q=merge_scores(root,ids,16)
            np.testing.assert_array_equal(p,np.arange(73))
            np.testing.assert_array_equal(q,-np.arange(73))
            np.savez(root/'test-rank-00.npz',positions=np.array([0,0]),source_ids=ids[[0,0]],
                truth_logits=np.zeros(2),generated_logits=np.zeros(2))
            with self.assertRaisesRegex(ValueError,'duplicate'):
                merge_scores(root,ids,16)

    def test_nersc_configuration(self):
        root=Path(__file__).resolve().parents[1]
        cfg=yaml.safe_load((root/'config/conditional_tau_weighted_audit.yaml').read_text())
        self.assertEqual(cfg['workers'],16)
        self.assertEqual(cfg['batch_size'],1024)
        self.assertEqual(cfg['source_run'],'443eg16h')
        self.assertEqual(cfg['arms'],list(ARMS))
        self.assertGreaterEqual(cfg['min_steps'],1000)

    def test_worker_cpu_smoke_without_starting_ray_cluster(self):
        import ray.train
        import ray.train.torch
        rng=np.random.default_rng(8)
        n=128
        a=dict(condition=rng.normal(size=(n,3)).astype('float32'),
            candidate_truth=rng.normal(size=(n,21)).astype('float32'),
            candidate_generated=rng.normal(size=(n,21)).astype('float32'),
            event_weight=np.ones(n),log_ratio=rng.normal(size=n),
            source_ids=np.arange(n).astype(str),split=np.repeat([0,1,2],[64,32,32]))
        ctx=Mock();ctx.get_world_rank.return_value=0;ctx.get_world_size.return_value=1
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);np.savez(root/'prepared.npz',**a)
            cfg=dict(prepared=str(root/'prepared.npz'),checkpoint=str(root/'best.pt'),
                arm='weighted_joint',seed=42,hidden=16,dropout=.05,relative_dim=6,
                batch_size=16,lr=.0002,min_lr=.00001,weight_decay=.001,epochs=3,
                min_steps=2,patience=10,min_delta=.0001,workers=1,bootstrap=20)
            with patch.object(ray.train,'get_context',return_value=ctx), \
                 patch.object(ray.train.torch,'get_device',return_value=torch.device('cpu')), \
                 patch.object(ray.train.torch,'prepare_model',side_effect=lambda m:m), \
                 patch.object(ray.train,'report') as log, \
                 patch('torch.distributed.all_reduce'),patch('torch.distributed.broadcast'), \
                 patch('torch.distributed.barrier'):
                worker(cfg)
            self.assertEqual(log.call_count,3)
            result=endpoint(cfg,a,'weighted_joint')
            self.assertEqual(result['fit_status']['steps'],12)
            self.assertTrue(np.isfinite(result['bce']))


if __name__=='__main__': unittest.main()
