import unittest
import tempfile
import copy
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path
import torch
from experiments.dgpo_toy.relational_retention import retention_contrasts, make_models, AnchoredBasisPolicy, default_plan
from experiments.dgpo_toy.relational_conditioning import verify_initial, encode_visible, RelationalData, RelationalCritic
from experiments.dgpo_toy.nonperiodic_cube import Data
from experiments.dgpo_toy import conditional as native
from experiments.dgpo_toy import closed_loop_lab as lab
from experiments.dgpo_toy.relational_directions import reward_gradient
from experiments.dgpo_toy.conditional_interference import flat_grad
from experiments.dgpo_toy.relational_refit import state_at_boundary, assert_replay


class RetentionTests(unittest.TestCase):
    def test_new_evaluation_seed_does_not_reuse_classifier_condition_stream(self):
        plan=default_plan(SimpleNamespace(output=lab.ARTIFACTS/'seed_test',steps=300,
            arms=['raw'],review_note='test',frozen_correction=False,
            initial_checkpoint=None,reward_directory=None,reward_key='raw__relative',
            anchored_basis=False))
        self.assertNotIn(plan['evaluation_seed'],
                         {plan['audit_panel_seed']+k for k in (10000,20000,30000)})

    def test_fixed_window_distinguishes_plateau_and_gain_erasure(self):
        torch.set_num_threads(1)
        base=torch.zeros(100,8)
        arrays={'baseline_0':base,'raw_5':base+.1,'raw_20':base+.01,
                'relative_5':base+.1,'relative_20':base+.11}
        result=retention_contrasts(arrays,['raw','relative'],100)
        self.assertTrue(result['failure_windows']['raw/5_to_20'])
        self.assertFalse(result['failure_windows']['relative/5_to_20'])
        arrays['raw_20']=base+.1
        self.assertFalse(retention_contrasts(arrays,['raw'],100)['any_failure'])

    def test_frozen_correction_preserves_initial_function(self):
        cfg=native.Config(dimensions=3,context_dim=1,hidden=8,ddim_steps=4)
        source=native.Denoiser(cfg)
        plan={'arms':['raw','relative'],'initialization_seed':17,'frozen_correction':True}
        models=make_models(source,cfg,plan)
        verify_initial(source,cfg,models)
        for model in models.values():
            model.requires_grad_(True)
            self.assertEqual(list(model.network.parameters()),[])
            self.assertTrue(list(model.condition_heads.parameters()))

    def test_refit_starts_from_exact_saved_policy(self):
        cfg=native.Config(dimensions=3,context_dim=1,hidden=8,ddim_steps=4)
        source=native.Denoiser(cfg)
        plan={'arms':['raw'],'initialization_seed':17}
        original=make_models(source,cfg,plan)['raw']
        with torch.no_grad():
            original.condition_heads[0].bias.add_(.12)
        with tempfile.TemporaryDirectory(dir=lab.ARTIFACTS) as folder:
            path=Path(folder)/'saved.pt'
            torch.save({'model':original.state_dict(),'representation':'raw'},path)
            loaded=make_models(source,cfg,{**plan,'initial_checkpoint':str(path)})['raw']
            for key,value in original.state_dict().items():
                self.assertTrue(torch.equal(value,loaded.state_dict()[key]))
            with self.assertRaises(ValueError):
                make_models(source,cfg,{**plan,'arms':['relative'],'initial_checkpoint':str(path)})

    def test_anchored_basis_function_and_raw_update_replay(self):
        torch.manual_seed(21)
        cfg=native.Config(dimensions=3,context_dim=1,hidden=8,ddim_steps=4,
            batch=8,candidates=4,timesteps=2,policy_steps=5,eval_events=8,eval_every=5)
        source=native.Denoiser(cfg)
        fitted=make_models(source,cfg,{'arms':['raw'],'initialization_seed':17})['raw']
        with torch.no_grad():
            for head in fitted.condition_heads:
                head.weight.normal_(0,.01);head.bias.normal_(0,.01)
        raw=AnchoredBasisPolicy(fitted,'raw')
        relative=AnchoredBasisPolicy(fitted,'relative')
        c=encode_visible(torch.rand(16,1)*1.9-.95,torch.rand(16,1)*6)
        x,t=torch.randn(16,3),torch.rand(16)
        for m in (raw,relative):
            self.assertTrue(torch.equal(m(x,t,c),fitted(x,t,c)))
            self.assertTrue(torch.equal(native.ddim(m,c,x,4),native.ddim(fitted,c,x,4)))
            m.requires_grad_(True)
            self.assertEqual(list(m.origin.parameters()),[])
            self.assertEqual(list(m.anchor.parameters()),[])
        reward=RelationalCritic(width=8).requires_grad_(False)
        data=RelationalData(Data(cfg))
        a,_=native.policy_train('dgpo',fitted,reward,data,cfg,17,31,lambda row:None,velocity_coefficient=1.)
        b,_=native.policy_train('dgpo',raw,reward,data,cfg,17,31,lambda row:None,velocity_coefficient=1.)
        torch.testing.assert_close(a(x,t,c),b(x,t,c),atol=2e-6,rtol=2e-5)

    def test_chunked_actual_reward_gradient_equals_direct_autograd(self):
        torch.manual_seed(17)
        cfg=native.Config(dimensions=3,context_dim=1,hidden=8,ddim_steps=4)
        source=native.Denoiser(cfg)
        m=make_models(source,cfg,{'arms':['raw'],'initialization_seed':17})['raw']
        critic=RelationalCritic(width=8).requires_grad_(False)
        data=RelationalData(Data(cfg));rng=native.generator(131)
        c=data.contexts(40,rng);z=torch.randn(40,4,3,generator=rng)
        halves=reward_gradient(m,critic,c,z,cfg)
        direct=flat_grad(critic(native.ddim(m,c[:,None],z,4),c[:,None]).mean(),m)
        torch.testing.assert_close(halves.mean(0),direct,atol=2e-7,rtol=3e-5)

    def test_refit_resume_keeps_optimizer_and_rng_and_checks_control(self):
        torch.set_num_threads(1)
        torch.manual_seed(19)
        cfg=native.Config(dimensions=3,context_dim=1,hidden=8,ddim_steps=4,
            batch=8,candidates=4,timesteps=2,policy_steps=2,eval_events=8,eval_every=1)
        initial=make_models(native.Denoiser(cfg),cfg,
                            {'arms':['raw'],'initialization_seed':17})['raw']
        reward=RelationalCritic(width=8).requires_grad_(False)
        data=RelationalData(Data(cfg));states={}
        def save(step,m,opt,rng,history):
            states[step]=state_at_boundary(m,opt,rng,history,{'test':True})
        native.policy_train('dgpo',initial,reward,data,cfg,17,31,lambda row:None,save,
                            velocity_coefficient=1.)
        expected=copy.deepcopy(states[2])
        native.policy_train('dgpo',initial,reward,data,cfg,17,31,lambda row:None,save,
                            resume_state=states[1],velocity_coefficient=1.)
        assert_replay(states[2],expected)
        bad=copy.deepcopy(expected);bad['rng'][0]^=1
        with self.assertRaises(AssertionError):
            assert_replay(bad,expected)
        refreshed=copy.deepcopy(initial);refreshed.load_state_dict(expected['model'])
        native.policy_train('dgpo',refreshed,reward,data,replace(cfg,policy_steps=3),
                            17,31,lambda row:None,save,resume_state=expected,
                            velocity_coefficient=1.)
        self.assertEqual(len(states[3]['history']),3)
        for state in states[3]['optimizer']['state'].values():
            self.assertEqual(float(state['step']),3.)
        self.assertFalse(torch.equal(states[3]['rng'],expected['rng']))


if __name__=='__main__':
    unittest.main()
