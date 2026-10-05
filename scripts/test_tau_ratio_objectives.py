"""CPU formula, DDP, inference-merge and matched-control tests; no remote jobs."""
import copy
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
import yaml

from scripts.tau_ratio_objectives import ratio_objective, validation_objective
from scripts.run_tau_ratio_objectives import (read_settings,check_protocol,make_arm_config,
    merge_scores,paired_cij_comparison,ROOT)


def ddp_check_worker(rank, rendezvous, output):
    from torch import distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2)
    try:
        # Unequal local counts. For c=.1, rank 0 alone would activate correction,
        # but the true global batch does not. c=.3 activates it globally.
        xp = torch.tensor([[np.log(100.),1.],[np.log(.1),1.],[np.log(.1),1.]],dtype=torch.float64)
        xq = torch.tensor([[np.log(.1),1.],[np.log(10.),1.],[np.log(10.),1.]],dtype=torch.float64)
        take = slice(0,1) if rank==0 else slice(1,3)
        records=[]
        for c in (.1,.3):
            model = nn.Linear(2,1,bias=False).double()
            with torch.no_grad():
                model.weight.copy_(torch.tensor([[1.,0.]],dtype=torch.float64))
            model=DDP(model)
            x=torch.cat((xp[take],xq[take]),0)
            pred=model(x).flatten()
            n=len(pred)//2
            loss,stats=ratio_objective(pred[:n],pred[n:],torch.ones(n), 'nnukl',c)
            loss.backward()
            records.append(dict(loss=loss.item(),active=stats['correction_active'].item(),
                                grad=model.module.weight.grad.tolist()))
        Path(output,f'rank{rank}.json').write_text(json.dumps(records))
    finally:
        dist.destroy_process_group()


class ObjectiveTests(unittest.TestCase):
    def test_mlc_formula_and_gradients(self):
        p=torch.tensor([.2,-.4],requires_grad=True)
        q=torch.tensor([.7,-.2],requires_grad=True)
        w=torch.tensor([.5,1.5])
        loss,stats=ratio_objective(p,q,w,'mlc',distributed=False)
        torch.testing.assert_close(loss,.5*(w.double()*(q.double().exp()-p)).mean())
        gp,gq=torch.autograd.grad(loss,(p,q))
        torch.testing.assert_close(gp,-.5*w/2)
        torch.testing.assert_close(gq,.5*w*q.exp()/2)
        self.assertEqual(stats['correction'].item(),0)

    def test_nnukl_inactive_equals_mlc_exactly(self):
        p=torch.tensor([.2,-.4],requires_grad=True)
        q=torch.tensor([.7,-.2],requires_grad=True)
        a,_=ratio_objective(p,q,torch.ones(2),'mlc',distributed=False)
        b,stats=ratio_objective(p,q,torch.ones(2),'nnukl',.01,distributed=False)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        ga=torch.autograd.grad(a,(p,q),retain_graph=True)
        gb=torch.autograd.grad(b,(p,q))
        for x,y in zip(ga,gb):
            torch.testing.assert_close(x,y,rtol=0,atol=0)
        self.assertEqual(stats['correction_active'].item(),0)

    def test_nnukl_active_matches_paper_and_gradients(self):
        p=torch.tensor([5.,6.],requires_grad=True)
        q=torch.tensor([0.,-1.],requires_grad=True)
        w=torch.ones(2); c=.1
        loss,stats=ratio_objective(p,q,w,'nnukl',c,distributed=False)
        self.assertEqual(stats['correction_active'].item(),1)
        expected=.5*((-p.double()+c*p.double().exp()).mean()+
                     torch.relu(q.double().exp().mean()-c*p.double().exp().mean()))
        torch.testing.assert_close(loss,expected)
        gp,gq=torch.autograd.grad(loss,(p,q))
        torch.testing.assert_close(gp,.5*(-1+c*p.exp())/2)
        torch.testing.assert_close(gq,torch.zeros_like(q),atol=1e-7,rtol=0)

    def test_weight_scale_not_minibatch_renormalized(self):
        p,q=torch.tensor([1.,2.]),torch.tensor([0.,1.])
        for kind,c in (('mlc',0),('nnukl',.1)):
            a,_=ratio_objective(p,q,torch.ones(2),kind,c,distributed=False)
            b,_=ratio_objective(p,q,torch.ones(2)*3,kind,c,distributed=False)
            torch.testing.assert_close(b,3*a)

    def test_zero_weights_and_overflow(self):
        p=torch.tensor([1000.,0.],requires_grad=True)
        q=torch.tensor([1000.,0.],requires_grad=True)
        loss,_=ratio_objective(p,q,torch.tensor([0.,1.]),'mlc',distributed=False)
        loss.backward()
        self.assertEqual(p.grad[0].item(),0)
        self.assertEqual(q.grad[0].item(),0)
        with self.assertRaises(FloatingPointError):
            ratio_objective(p,q,torch.ones(2),'mlc',distributed=False)
        for kind,c in (('nnukl',0),('nnukl',1),('mlc',.1)):
            with self.assertRaises(ValueError):
                ratio_objective(p,q,torch.ones(2),kind,c,distributed=False)

    def test_true_ratio_population_stationary_when_bound_holds(self):
        # Two-point distributions, true max p/q=2; c=.1 obeys the bound.
        den=torch.tensor([.25,.75],dtype=torch.float64)
        num=torch.tensor([.5,.5],dtype=torch.float64)
        s=(num/den).log().requires_grad_()
        # Enumerate both categories with separate class measures using repeated
        # sample pairs: positive [0,0,1,1], negative [0,1,1,1].
        loss,stats=ratio_objective(s[[0,0,1,1]],s[[0,1,1,1]],torch.ones(4),
                                  'nnukl',.1,distributed=False)
        loss.backward()
        torch.testing.assert_close(s.grad,torch.zeros_like(s),atol=1e-14,rtol=0)
        self.assertEqual(stats['correction_active'].item(),0)

    def test_validation_formula(self):
        v=validation_objective(np.array([1.,2.]),np.array([0.,1.]),np.array([2.,4.]),'nnukl',.01)
        self.assertTrue(all(np.isfinite(value) for value in v.values()))
        self.assertGreater(v['paired_bce'],0)

    def test_real_two_rank_ddp_matches_single_global_batch(self):
        import torch.multiprocessing as mp
        with tempfile.TemporaryDirectory() as d:
            mp.spawn(ddp_check_worker,args=(str(Path(d)/'rendezvous'),d),nprocs=2,join=True)
            xp=torch.tensor([[np.log(100.),1.],[np.log(.1),1.],[np.log(.1),1.]],dtype=torch.float64)
            xq=torch.tensor([[np.log(.1),1.],[np.log(10.),1.],[np.log(10.),1.]],dtype=torch.float64)
            for index,c in enumerate((.1,.3)):
                model=nn.Linear(2,1,bias=False).double()
                with torch.no_grad():
                    model.weight.copy_(torch.tensor([[1.,0.]],dtype=torch.float64))
                loss,stats=ratio_objective(model(xp).flatten(),model(xq).flatten(),torch.ones(3),
                                          'nnukl',c,distributed=False)
                loss.backward()
                for rank in range(2):
                    actual=json.loads(Path(d,f'rank{rank}.json').read_text())[index]
                    self.assertAlmostEqual(actual['loss'],loss.item(),places=12)
                    self.assertEqual(actual['active'],stats['correction_active'].item())
                    np.testing.assert_allclose(actual['grad'],model.weight.grad.numpy(),atol=1e-12,rtol=1e-12)


class ProtocolTests(unittest.TestCase):
    def test_all_attempts_only_new_arms_even_if_mlc_fails(self):
        from scripts.run_tau_ratio_objectives import main
        args=['runner',str(ROOT/'config/conditional_tau_ratio_objectives_10pct.yaml'),'all']
        with patch('sys.argv',args),patch('subprocess.run',side_effect=[
            SimpleNamespace(returncode=1),SimpleNamespace(returncode=0)]) as run:
            with self.assertRaisesRegex(SystemExit,'mlc'):
                main()
        self.assertEqual(run.call_count,2)
        self.assertEqual([call.args[0][3] for call in run.call_args_list],['mlc','nnukl'])

    def test_early_stop_counts_after_gate_but_saves_tiny_improvements(self):
        import ray.train.torch
        from scripts.train_conditional_spin_ratio import training_worker
        rng=np.random.default_rng(9)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            np.savez(root/'prepared.npz',condition=rng.normal(size=(24,5)).astype('float32'),
                candidate_truth=rng.normal(size=(24,21)).astype('float32'),
                candidate_generated=rng.normal(size=(24,21)).astype('float32'),
                event_weight=np.ones(24,dtype='float32'),split=np.array([0]*16+[1]*8),
                condition_mean=np.zeros(5),condition_scale=np.ones(5))
            for minimum,expected_epochs in ((0,3),(6,4)):
                cfg=dict(seed=42,prepared=str(root/'prepared.npz'),checkpoint=str(root/'best.pt'),
                    batch_size=8,hidden=16,dropout=0.,relative_dim=6,
                    lr=2e-4,min_lr=1e-5,weight_decay=.001,epochs=6,min_delta=.001,
                    min_steps=minimum,patience=2,representation='tau',packing_spec={},
                    condition_normalization='masked_feature',mmd_coefficient=0.,
                    ratio_objective='nnukl',nnukl_c=.01)
                reports=[]
                metric_sequence=[dict(bce=.5-i*.00001,auc=.7) for i in range(6)]
                with patch('ray.train.get_context',return_value=SimpleNamespace(
                        get_world_rank=lambda:0,get_world_size=lambda:1)), \
                     patch('ray.train.torch.get_device',return_value=torch.device('cpu')), \
                     patch('ray.train.torch.prepare_model',side_effect=lambda x:x), \
                     patch('ray.train.report',side_effect=reports.append), \
                     patch('torch.distributed.all_reduce'),patch('torch.distributed.broadcast'), \
                     patch('scripts.train_conditional_spin_ratio.pair_metrics',side_effect=metric_sequence):
                    training_worker(cfg)
                self.assertEqual(len(reports),expected_epochs)
                self.assertEqual(reports[-1]['early_stop_triggered'],1)
                saved=torch.load(cfg['checkpoint'],weights_only=True)
                self.assertEqual(saved['epoch'],expected_epochs-1)
                self.assertEqual(saved['val_bce'],metric_sequence[expected_epochs-1]['bce'])

    def test_actual_training_worker_smoke_for_each_objective(self):
        # Exercise the shared epoch/checkpoint/report path without a Ray cluster.
        import ray.train.torch
        from scripts.train_conditional_spin_ratio import training_worker
        rng=np.random.default_rng(22)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            np.savez(root/'prepared.npz',condition=rng.normal(size=(48,5)).astype('float32'),
                candidate_truth=rng.normal(size=(48,29)).astype('float32'),
                candidate_generated=rng.normal(size=(48,29)).astype('float32'),
                event_weight=np.ones(48,dtype='float32'),split=np.array([0]*32+[1]*16),
                condition_mean=np.zeros(5),condition_scale=np.ones(5))
            for kind,c in (('bce',0),('mlc',0),('nnukl',.01)):
                cfg=dict(seed=42,prepared=str(root/'prepared.npz'),checkpoint=str(root/f'{kind}.pt'),
                    batch_size=8,hidden=16,dropout=.05,relative_dim=6,backbone_cache='frozen',
                    lr=2e-4,min_lr=1e-5,weight_decay=.001,epochs=2,min_delta=0.,min_steps=0,patience=3,
                    representation='tau',packing_spec={},condition_normalization='masked_feature',
                    mmd_coefficient=0.,ratio_objective=kind,nnukl_c=c)
                ctx=SimpleNamespace(get_world_rank=lambda:0,get_world_size=lambda:1)
                reports=[]
                with patch('ray.train.get_context',return_value=ctx), \
                     patch('ray.train.torch.get_device',return_value=torch.device('cpu')), \
                     patch('ray.train.torch.prepare_model',side_effect=lambda x:x), \
                     patch('ray.train.report',side_effect=reports.append), \
                     patch('torch.distributed.all_reduce'),patch('torch.distributed.broadcast'):
                    training_worker(cfg)
                self.assertEqual(len(reports),2)
                self.assertEqual(reports[-1]['optimizer_steps'],8)
                saved=torch.load(cfg['checkpoint'],weights_only=True)
                self.assertEqual(saved['ratio_objective'],kind)
                if kind!='bce':
                    self.assertIn('ratio_train/correction_active_fraction',reports[-1])
                    self.assertIn('ratio_val/objective',reports[-1])
                    self.assertTrue(all(np.isfinite(v) for v in reports[-1].values()))

    def config_and_baseline(self):
        cfg=read_settings(ROOT/'config/conditional_tau_ratio_objectives_10pct.yaml')
        expected=yaml.safe_load((ROOT/cfg['baseline_config']).read_text())
        baseline=dict(expected['classifier'],representation='tau',relative_dim=6,
            candidate_count=1,weight_mode='joint',train_events=expected['platform']['data_parquet_dir'],
            test_events=expected['platform']['data_parquet_val_dir'],
            train_source=expected['experiment']['train_sample_root'],test_source=expected['experiment']['test_source'],
            train_manifest={'weights':'raw_state_dict_only'},backbone_manifest={'global_step':1110})
        return cfg,baseline,expected

    def test_reuses_bce_and_changes_only_objective(self):
        cfg,baseline,expected=self.config_and_baseline()
        check_protocol(cfg,baseline,expected)
        a=make_arm_config(cfg,baseline,Path(cfg['baseline_directory']),Path('/tmp/mlc'),'mlc')
        b=make_arm_config(cfg,baseline,Path(cfg['baseline_directory']),Path('/tmp/nnukl'),'nnukl')
        for key in expected['classifier']:
            self.assertEqual(a[key],b[key])
        self.assertTrue(a['no_bce_refit'])
        self.assertEqual(a['nnukl_c'],0)
        self.assertEqual(b['nnukl_c'],.01)
        self.assertEqual(a['baseline_run'],'443eg16h')
        self.assertEqual(a['patience'],25)
        self.assertEqual(a['min_delta'],.0001)
        self.assertEqual(a['min_steps'],1000)
        self.assertEqual(a['baseline_training_differences']['patience'],dict(baseline=251,current=25))
        with self.assertRaisesRegex(ValueError,'BCE is reused'):
            make_arm_config(cfg,baseline,Path('/tmp/a'),Path('/tmp/b'),'bce')
        for key,value in (('batch_size',256),('relative_dim',0),('mmd_coefficient',1)):
            bad=copy.deepcopy(baseline);bad[key]=value
            with self.assertRaises(ValueError):
                check_protocol(cfg,bad,expected)

    def test_shard_merge_16_no_padding(self):
        with tempfile.TemporaryDirectory() as d:
            ids=np.array([f'event-{i}' for i in range(53)])
            for rank in range(16):
                idx=np.arange(rank,len(ids),16)
                np.savez(Path(d)/f'score-{rank:02d}.npz',positions=idx,source_ids=ids[idx],
                         truth_logits=idx+.5,generated_logits=-idx)
            p,q=merge_scores(d,ids,16)
            np.testing.assert_equal(p,np.arange(53)+.5)
            np.testing.assert_equal(q,-np.arange(53))
            np.savez(Path(d)/'score-00.npz',positions=[1],source_ids=ids[[1]],
                     truth_logits=[0.],generated_logits=[0.])
            with self.assertRaisesRegex(ValueError,'Missing, duplicate'):
                merge_scores(d,ids,16)

    def test_cij_same_estimator_and_paired_comparison(self):
        from scripts.diagnose_reweighted_cij import analyze
        rng=np.random.default_rng(1);n=80
        def unit():
            x=rng.normal(size=(n,3));return x/np.linalg.norm(x,axis=1)[:,None]
        a=dict(split=np.full(n,2),truth_a=unit(),truth_b=unit(),sample_a=unit(),sample_b=unit(),
               kappas=np.tile([1.,-.33],(n,1)),event_weight=np.exp(rng.normal(size=n)))
        logits=rng.normal(size=n)
        report=paired_cij_comparison(a,dict(bce=logits,candidate_ratio=logits),20,42)
        data=dict(event_id=np.arange(n).astype(str),truth_a=a['truth_a'],truth_b=a['truth_b'],
            sample_a=a['sample_a'][:,None],sample_b=a['sample_b'][:,None],
            log_ratio=logits[:,None],event_weight=a['event_weight'])
        old=analyze(data,a['kappas'],'joint',20,42)
        np.testing.assert_allclose(report['arms']['candidate_ratio']['C'],old['results']['reweighted']['C'],atol=1e-13)
        same=report['comparisons']['candidate_minus_bce']
        self.assertEqual(same['value'],0)
        np.testing.assert_allclose(same['ci95'],[0,0],atol=1e-13)
        np.testing.assert_allclose(report['comparisons']['candidate_minus_unweighted']['ci95'],
            old['comparison']['C_frobenius_error_change']['ci95'],atol=1e-12)


if __name__=='__main__':
    unittest.main()
