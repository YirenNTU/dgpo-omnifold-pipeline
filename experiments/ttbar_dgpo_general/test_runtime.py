"""CPU scientific/integration checks; no NERSC job or policy training."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
import yaml
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
EVE=Path(os.environ.get('EVENET_ROOT','/Users/yirenwu/EveNet-private'));sys.path.insert(0,str(EVE))
from experiments.ttbar_dgpo_general.reward import *
from experiments.ttbar_dgpo_general.runtime import validate_settings,Session
from experiments.ttbar_general_classifier.core import geometry,mixture,top_oracle_inputs
from experiments.ttbar_general_classifier.run import extract_features,observed_condition,tensor_batch,parts,write_json,atomic_torch
from experiments.ttbar_general_classifier.test_pipeline import extras

class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_numpy_parity_and_loo(self):
        from RL.DGPO_neutrino.dgpo_utils import compute_per_event_advantage
        rng=np.random.default_rng(42);b,k=11,8
        truth=rng.normal(size=(b,2,3));truth[...,0]=3.+abs(truth[...,0])
        x=truth[:,None]+rng.normal(scale=.15,size=(b,k,2,3));logits=rng.normal(size=(b,k));scale=[150,70,70,70]
        p,t=top_oracle_inputs(x,extras(b),truth,'reco_top_truthnu_p4')
        for alpha in (0,.3,.5):
            g,m,d=mixture(logits,p,t,scale,alpha=alpha)
            logw,logq,dist=log_mixture(torch.tensor(logits.T),torch.tensor(x.transpose(1,0,2,3)),torch.tensor(truth),scale,alpha)
            np.testing.assert_allclose(logw.exp().numpy().T,m,atol=1e-12)
            np.testing.assert_allclose(dist.numpy().T,d,atol=1e-12)
            adv,_=compute_per_event_advantage(logw,estimator='leave_one_out_unscaled')
            torch.testing.assert_close(adv.sum(0),torch.zeros(b,dtype=adv.dtype),atol=1e-12,rtol=0)
            if alpha==0:
                expected,_=compute_per_event_advantage(torch.tensor(logits.T),estimator='leave_one_out_unscaled')
                torch.testing.assert_close(adv,expected)
        np.testing.assert_allclose(geometry(x),geometry_tensor(torch.tensor(x)).numpy(),atol=1e-7)
        # Stable at extreme classifier scores; no additive reward approximation.
        logw,_,_=log_mixture(torch.tensor(logits.T)*1e4,torch.tensor(x.transpose(1,0,2,3)),torch.tensor(truth),scale)
        self.assertTrue(torch.isfinite(logw).all())
        torch.testing.assert_close(logw.exp().sum(0),torch.ones(b,dtype=logw.dtype))

    def test_saved_classifier_cache_parity(self):
        folder=ROOT/'artifacts/ttbar_general_oracle/frozen-general-oracle30-mixture-svd-run-01'
        if not (folder/'classifier-best.pt').exists():self.skipTest('Saved experiment not on this machine')
        payload=torch.load(folder/'classifier-best.pt',map_location='cpu',weights_only=False);st=payload['stats']
        head=GeneralClassifier(st['condition_dim'],st['candidate_dim'],**payload['config']['head']).eval()
        head.load_state_dict(payload['state_dict']);a=next(parts(folder,'validation'))[2];n=16;k=8
        f=torch.tensor(np.array(a['generated_features'][:n]));co=torch.tensor(np.array(a['condition'][:n]));la=torch.tensor(np.array(a['labels'][:n]))
        with torch.no_grad():logits=head(co.repeat_interleave(k,0),((f-st['mean'])/st['scale']).reshape(n*k,-1),la.repeat_interleave(k)).reshape(n,k)
        p,t=top_oracle_inputs(a['candidates'][:n],{key:np.asarray(v[:n]) for key,v in a.items() if key.startswith('EXTRA/')},a['truth'][:n],'reco_top_truthnu_p4')
        expected=mixture(logits.numpy(),p,t,st['oracle_scale'])[1]
        actual,_,_=log_mixture(logits.T,torch.tensor(np.array(a['candidates'][:n])).permute(1,0,2,3),torch.tensor(np.array(a['truth'][:n])),st['oracle_scale'])
        np.testing.assert_allclose(actual.exp().numpy().T,expected,rtol=2e-6,atol=2e-7)
        from types import SimpleNamespace
        from experiments.ttbar_dgpo_general.validation import summarize
        with tempfile.TemporaryDirectory() as tmp:
            part=next(parts(folder,'validation'))[0]
            write_json(Path(tmp)/'CACHE_COMPLETE.json',{'parts':[str(part)]})
            reward=SimpleNamespace(device=torch.device('cpu'),head=head,mean=st['mean'],scale=st['scale'],stats=st,alpha=.3,temperature=1.)
            report=summarize(tmp,reward)
            self.assertEqual(report['val/events'],len(a['truth']))
            self.assertTrue((Path(tmp)/'distributions.png').exists())
            self.assertIn('val/distribution/top1_mass_reco_truthnu/TV',report)

    def test_frozen_torch_features_and_candidate_chunking(self):
        spec=importlib.util.spec_from_file_location('fixture',EVE/'tests/test_ttbar_visible_film.py');f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
        cfg,norm=f.config_and_norm()
        from evenet.network.evenet_model import build_evenet_model_from_training_config
        model=build_evenet_model_from_training_config(cfg,norm,torch.device('cpu')).eval().requires_grad_(False)
        a={k:v.numpy() for k,v in f.batch().items()};a['classification']=np.zeros(2);a['num_sequential_vectors']=np.array([4,4]);b=tensor_batch(a,'cpu')
        co=observed_condition(model,b);ft=extract_features(model,b,b['x_invisible'])
        np.testing.assert_allclose(condition_tensor(model,b).numpy(),co,atol=1e-7)
        np.testing.assert_allclose(features_tensor(model,b,b['x_invisible']).numpy(),ft,atol=2e-6)
        st=dict(mean=torch.zeros(ft.shape[1]),scale=torch.ones(ft.shape[1]),condition_dim=co.shape[1],candidate_dim=ft.shape[1],oracle_reference='reco_top_truthnu_p4',oracle_scale=[100,50,50,50])
        head=GeneralClassifier(co.shape[1],ft.shape[1]);payload=dict(stats=st,state_dict=head.state_dict(),config={'head':{}})
        x=b['x_invisible'][None].repeat(8,1,1,1)+torch.randn(8,2,2,3)*.02
        r=GeneralOracleReward(model,payload,feature_batch_size=2);a=r.compute(x,b)
        r.feature_batch_size=16;z=r.compute(x,b)
        torch.testing.assert_close(a,z,atol=2e-6,rtol=2e-6)
        self.assertFalse(a.requires_grad);self.assertTrue(all(p.grad is None for p in model.parameters()))
        poisoned={**b,'x_invisible':b['x_invisible']+1}
        _=r.compute(x,poisoned);score=r.last['logits'].clone();_=r.compute(x,b)
        torch.testing.assert_close(score,r.last['logits']) # MC truth cannot enter General

        # One tiny CPU update through the ACTUAL native optimizer-facing DGPO path.
        # The feature backbone stays frozen while a separate policy is optimized.
        sys.path.insert(0,str(EVE))
        from RL.DGPO_neutrino import dgpo_trainer as native
        from RL.DGPO_neutrino.rewards import RewardAggregator
        from evenet.control.global_config import global_config,DotDict
        from evenet.utilities.diffusion_sampler import DDIMSampler
        policy=build_evenet_model_from_training_config(cfg,norm,torch.device('cpu'))
        policy.load_state_dict(model.state_dict());reference=build_evenet_model_from_training_config(cfg,norm,torch.device('cpu'))
        reference.load_state_dict(model.state_dict());reference.eval().requires_grad_(False)
        agg=RewardAggregator();agg.add(r,1.)
        previous=global_config._global_config
        global_config._global_config=copy.deepcopy(cfg._global_config)
        global_config._global_config['dgpo']=DotDict({'projection_constraint':{'enabled':False}})
        try:
            for settings in global_config.options.Training.Components.values():
                if isinstance(settings,dict):settings['warm_up']=False
            opt=native.build_optimizer(policy,steps_per_epoch=1,warmup_steps=1,is_rank0=False)
            before={k:v.clone() for k,v in policy.state_dict().items()}
            result=native.train_step(policy,reference,None,None,b,opt,DDIMSampler(torch.device('cpu')),agg,
                beta=1.,K=8,num_ddim_steps=3,global_step=0,epoch=0,device=torch.device('cpu'),dtype=torch.float32,
                round_ref_model=reference,reference_trust_coefficient=1.,num_train_timesteps=2,
                advantage_estimator='leave_one_out_unscaled',rollout_chunk_k=2)
            self.assertTrue(np.isfinite(result['train/loss/total']))
            self.assertTrue(any(not torch.equal(before[k],v) for k,v in policy.state_dict().items()))
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            from RL.DGPO_neutrino.model_utils import save_lightning_compatible_checkpoint,state_dict_sha256,assert_round_reference_pairing
            session=Session.__new__(Session);session.reward=r;session.contract={'test':True};session.refit_step=500;session.round_id=1;session.reference_sha=state_dict_sha256(reference)
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'last.ckpt'
                save_lightning_compatible_checkpoint(path,policy,None,global_config,last_completed_epoch=1,dgpo_next_epoch=2,
                    global_step=510,optimizer=opt,ref_model=reference,round_ref_model=reference,reward_round_id=1,
                    dgpo_omnifold_reward_stack=session.stack())
                saved=torch.load(path,map_location='cpu',weights_only=False)
                self.assertEqual(saved['dgpo_omnifold_reward_stack']['refit_step'],500)
                self.assertEqual(saved['dgpo_omnifold_reward_stack']['alpha'],.3)
                self.assertIn('dgpo_optimizer_state_dict',saved)
                assert_round_reference_pairing(reference,reward_round_id=1,reward_reference_sha256=saved['dgpo_omnifold_reward_stack']['reference_sha'],where='CPU regression')
                with torch.no_grad():next(reference.parameters()).add_(.1)
                with self.assertRaisesRegex(RuntimeError,'mismatch'):
                    assert_round_reference_pairing(reference,reward_round_id=1,reward_reference_sha256=saved['dgpo_omnifold_reward_stack']['reference_sha'],where='CPU regression')
        finally:global_config._global_config=previous

    def test_config_contract(self):
        c=yaml.safe_load((ROOT/'config/ttbar_dgpo_general/general70_top_oracle30_full_mg5.yaml').read_text());validate_settings(c)
        for key in ('workers','batch_size','candidates'):
            wrong=copy.deepcopy(c);wrong[key]=1
            with self.assertRaises(ValueError):validate_settings(wrong)

    def test_wandb_stable_id_and_explicit_resume(self):
        from experiments.ttbar_dgpo_general.runtime import start_wandb
        from evenet.control.global_config import global_config
        c=yaml.safe_load((ROOT/'config/ttbar_dgpo_general/general70_top_oracle30_full_mg5.yaml').read_text())
        with patch('wandb.init') as init,patch('wandb.define_metric'),patch.object(global_config,'to_logger',return_value={}):
            start_wandb(c)
            self.assertEqual(init.call_args.kwargs['id'],c['wandb']['id'])
            self.assertEqual(init.call_args.kwargs['name'],c['wandb']['run_name'])
            self.assertEqual(init.call_args.kwargs['group'],c['wandb']['group'])
            self.assertEqual(init.call_args.kwargs['resume'],'never')
            start_wandb(c,resume_run_id=c['wandb']['id']);self.assertEqual(init.call_args.kwargs['resume'],'must')
            with self.assertRaises(ValueError):start_wandb(c,resume_run_id='another-run')

    def test_refit_installs_only_after_plateau_and_recenters_reference(self):
        c=yaml.safe_load((ROOT/'config/ttbar_dgpo_general/general70_top_oracle30_full_mg5.yaml').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            c['output']=tmp;s=Session.__new__(Session);s.c=c;s.rank=0;s.world=1;s.device=torch.device('cpu');s.contract={'test':True};s.refit_step=0;s.round_id=0
            model=torch.nn.Linear(2,2);ref=torch.nn.Linear(2,2)
            from types import SimpleNamespace
            head=torch.nn.Linear(2,1)
            s.reward=SimpleNamespace(head=head,stats={'normalization':'fixed'},feature_model=model,
                install=lambda state:head.load_state_dict(state))
            s.agg=SimpleNamespace(_reference_sha=None)
            def pool(c,path,*args,**kwargs):
                return {'output':str(path),'fit':{}}
            def fit(cfg,**kwargs):
                out=Path(cfg['output']);atomic_torch(out/'classifier-best.pt',{'state_dict':kwargs['initial_state']})
                write_json(out/'FIT_COMPLETE.json',dict(plateau=True,best_validation_bce=.65))
            with patch('experiments.ttbar_dgpo_general.runtime.make_pool',side_effect=pool),patch('experiments.ttbar_dgpo_general.runtime.classifier.fit_worker',side_effect=fit),patch('experiments.ttbar_dgpo_general.runtime.barrier'):
                s.maybe_refit(model,ref,499);self.assertEqual(s.round_id,0)
                s.maybe_refit(model,ref,500);self.assertEqual(s.round_id,1);self.assertEqual(s.refit_step,500)
                torch.testing.assert_close(model.weight,ref.weight);self.assertFalse(ref.weight.requires_grad)
                s.maybe_refit(model,ref,501);self.assertEqual(s.round_id,1)
                # Reaching a fit cap does NOT accept the head/reference.
                bad=Path(tmp)/'refits/step-001000';bad.mkdir(parents=True)
                write_json(bad/'FIT_COMPLETE.json',dict(plateau=False))
                with self.assertRaisesRegex(RuntimeError,'plateau'):s.maybe_refit(model,ref,1000)
                self.assertEqual(s.round_id,1)

if __name__=='__main__':unittest.main()
