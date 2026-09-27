import copy
from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.dgpo_toy import closed_loop_lab as lab
from experiments.dgpo_toy import conditional as native
from experiments.dgpo_toy import relational_experiment as experiment
from experiments.dgpo_toy.nonperiodic_cube import Data, panel
from experiments.dgpo_toy.cube_swap import modes
from experiments.dgpo_toy.truth_pretrain import atomic_checkpoint, atomic_json
from experiments.dgpo_toy.relational_conditioning import (
    REPRESENTATIONS, RelationalCritic, RelationalData, RelationalPolicy, VisibleSource,
    condition_features, decode_condition, encode_visible, lift_panels, rotate_visible, verify_initial)


class RelationalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def cfg(self):
        return native.Config(dimensions=3,context_dim=1,hidden=8,policy_steps=4,
            batch=4,candidates=3,timesteps=2,ddim_steps=4,eval_events=8,eval_every=2)

    def visible(self,n=32):
        c = torch.linspace(-.98,.98,n)[:,None]
        return c,encode_visible(c,torch.linspace(-3,3,n)[:,None])

    def test_geometry_recovery_and_rotation(self):
        c,v = self.visible()
        torch.testing.assert_close(decode_condition(v),c,atol=2e-7,rtol=0)
        vv = rotate_visible(v,1.37)
        torch.testing.assert_close(decode_condition(vv),c,atol=2e-7,rtol=0)
        f,g = condition_features(v,'relative'),condition_features(vv,'relative')
        torch.testing.assert_close(f[:,4:],g[:,4:],atol=2e-6,rtol=0)
        self.assertGreater(float((condition_features(v,'particle')[:,4:]-
                                  condition_features(vv,'particle')[:,4:]).abs().max()),.1)

    def test_harmonics_and_equal_rms(self):
        c,v = self.visible()
        f = condition_features(v,'relative')[:,4:]
        expected = torch.stack([fn(k*math.pi*c[:,0]) for k in range(1,5)
                                for fn in (torch.sin,torch.cos)],-1).repeat_interleave(2,-1)
        torch.testing.assert_close(f,expected,atol=2e-6,rtol=0)
        for rep in ('particle','relative'):
            rms2 = condition_features(v,rep)[:,4:].square().mean(-1)
            torch.testing.assert_close(rms2,torch.full_like(rms2,.5),atol=1e-6,rtol=0)

    def test_classifier_slots_y_fixed_and_no_scalar_api(self):
        c,v = self.visible(); y = torch.randn(32,3)
        fs = {r:RelationalCritic(r).features(y,v) for r in REPRESENTATIONS}
        self.assertEqual(fs['raw'].shape,(32,47))
        self.assertEqual(float(fs['raw'][:,-16:].abs().sum()),0)
        for rep in REPRESENTATIONS:
            self.assertTrue(torch.equal(fs['raw'][:,:31],fs[rep][:,:31]))
        with self.assertRaises((ValueError,RuntimeError)):
            condition_features(c,'raw')
        # Scalar broadcasting must not silently masquerade as four visible inputs.
        with self.assertRaises(ValueError):
            RelationalCritic()(y,c)

    def test_lifting_keeps_archived_labels_and_global_rng(self):
        data = Data(self.cfg())
        pp = {s:panel(data,32,i) for s,i in [('train',1),('validation',2),('test',3)]}
        before = torch.get_rng_state().clone()
        lifted = lift_panels(pp,193)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        for split in pp:
            for key in ('positive','negative'):
                self.assertTrue(torch.equal(pp[split][key],lifted[split][key]))
            self.assertTrue(torch.equal(lifted[split]['scalar_c'],pp[split]['c']))
            self.assertEqual(lifted[split]['c'].shape,(32,4))

    def test_checked_lift_rejects_nonfinite_and_misaligned_data(self):
        data = Data(self.cfg())
        pp = {s:panel(data,32,i) for s,i in [('train',1),('validation',2),('test',3)]}
        good = experiment.checked_lift(pp,193)
        self.assertTrue(torch.equal(good['test']['positive'],pp['test']['positive']))
        for key in ('c', 'positive', 'negative'):
            broken = copy.deepcopy(pp); broken['train'][key][0,0] = float('nan')
            with self.assertRaises(ValueError):
                experiment.checked_lift(broken,193)
        broken = copy.deepcopy(pp); broken['test']['negative'] = broken['test']['negative'][:-1]
        with self.assertRaises(ValueError):
            experiment.checked_lift(broken,193)
        broken = copy.deepcopy(pp); broken['test']['c'][0] = 1.2
        with self.assertRaises(ValueError):
            experiment.checked_lift(broken,193)

    def test_policy_initial_matching_and_buffers_cannot_unfreeze(self):
        cfg = self.cfg(); source = native.Denoiser(cfg)
        models = experiment.make_policies(source,cfg,'relative',31)
        report = verify_initial(source,cfg,models)
        self.assertTrue(all(all(v.values()) for v in report['arms'].values()))
        self.assertEqual(sum(p.numel() for p in models['raw'].parameters()),
                         sum(p.numel() for p in models['relative'].parameters()))
        models['raw'].requires_grad_(True)
        self.assertEqual(list(models['raw'].base.parameters()),[])
        self.assertEqual(list(models['raw'].anchor.parameters()),[])
        self.assertTrue(all(not b.requires_grad for b in models['raw'].base.buffers()))

    def test_correction_does_not_decode_c(self):
        cfg = self.cfg(); source = native.Denoiser(cfg)
        model = RelationalPolicy(source,cfg,'raw')
        _,v = self.visible(8)
        with patch('experiments.dgpo_toy.relational_conditioning.decode_condition',side_effect=AssertionError('leak')):
            delta,_,_ = model.correction(torch.randn(8,3),torch.rand(8),v)
            self.assertTrue(torch.equal(delta,torch.zeros_like(delta)))

    def test_native_loss_exact_resume_and_frozen_anchor(self):
        cfg = self.cfg(); source = native.Denoiser(cfg)
        model = RelationalPolicy(source,cfg,'relative',width=8)
        data = RelationalData(Data(cfg))
        reward = RelationalCritic(width=8).requires_grad_(False)
        saved = {}
        def checkpoint(step,m,opt,rng,history):
            saved.update(model=copy.deepcopy(m.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),
                rng=rng.get_state(),history=copy.deepcopy(history),step=step,velocity_coefficient=1.)
        before = torch.get_rng_state().clone()
        full,_ = native.policy_train('dgpo',model,reward,data,cfg,21,93,lambda row:None,velocity_coefficient=1.)
        native.policy_train('dgpo',model,reward,data,replace(cfg,policy_steps=2),21,93,lambda row:None,
                            checkpoint,velocity_coefficient=1.)
        resumed,history = native.policy_train('dgpo',model,reward,data,cfg,21,93,lambda row:None,
                            resume_state=saved,velocity_coefficient=1.)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        for name,tensor in full.state_dict().items():
            self.assertTrue(torch.equal(tensor,resumed.state_dict()[name]),name)
            if name.startswith(('base.','anchor.')):
                self.assertTrue(torch.equal(tensor,model.state_dict()[name]),name)
        self.assertEqual(len(history),4)
        self.assertIn('main_velocity_gradient_cosine',history[-1])
        self.assertTrue(all(p.grad is None for p in reward.parameters()))

    def test_generic_fit_with_relational_factory_and_exact_selection(self):
        data = Data(self.cfg())
        pp = lift_panels({s:panel(data,32,i) for s,i in [('train',1),('validation',2),('test',3)]},13)
        with tempfile.TemporaryDirectory(dir=lab.ARTIFACTS,prefix='relation_test_') as tmp:
            report,_ = lab.fit_classifiers({'source':pp},REPRESENTATIONS,Path(tmp)/'fits',lambda row:None,
                model_factory=RelationalCritic,width=8,max_steps=2,min_steps=1,check_every=1)
            self.assertEqual(report['model_type'],'RelationalCritic')
            self.assertEqual(report['state'],'inconclusive_fit_budget')
            for key,value in report['test'].items():
                expected = min((r for r in report['history'] if r['arm']==key),key=lambda r:r['validation_bce'])
                self.assertEqual(value['selected_step'],expected['step'])
                model,loaded = experiment.load_critic(Path(tmp)/'fits',key)
                self.assertEqual(model.feature_mode,key.split('__')[1])
                self.assertFalse(loaded['valid'])

    def test_plan_scope_and_review_requirement(self):
        folder = Path(__file__).parent/'plans'
        for name in ('classifier','rl'):
            plan = json.loads((folder/f'relational_{name}.json').read_text())
            experiment.validate_plan(plan)
        with self.assertRaises(ValueError):
            experiment.artifact('/tmp/relation_outside_toy')
        with self.assertRaises(ValueError):
            experiment.rl_stage(plan,None,None,None,'')
        broken = copy.deepcopy(plan); broken['velocity_coefficient'] = 0
        with self.assertRaises(ValueError):
            experiment.validate_plan(broken)
        for field, value in (('width',32), ('min_steps',8000)):
            broken = copy.deepcopy(plan); broken['fit'][field] = value
            with self.assertRaises(ValueError):
                experiment.validate_plan(broken)
        broken = copy.deepcopy(plan); broken['wandb_mode'] = 'online'
        with self.assertRaises(ValueError):
            experiment.validate_plan(broken)

    def test_resolved_setup_has_actual_fit_and_policy_clocks(self):
        plan = json.loads((Path(__file__).parent/'plans/relational_rl.json').read_text())
        setup = experiment.resolved_setup(plan,self.cfg(),'disabled','relative')
        self.assertEqual(setup['classifier']['lr'],3e-4)
        self.assertEqual(setup['classifier']['paired_contexts_per_batch'],256)
        self.assertEqual(setup['classifier']['batch_examples'],512)
        self.assertEqual(setup['classifier']['min_steps'],32000)
        self.assertEqual(setup['policy']['lr'],1e-4)
        self.assertEqual(setup['policy']['audit_steps'],[0,25,100,300,1000])
        self.assertTrue(setup['policy']['same_frozen_reward_for_both_arms'])
        self.assertFalse(setup['next_stage_automatic'])
        self.assertEqual(setup['wandb']['mode'],'disabled')

    def test_nonplateau_never_eligible_for_rl(self):
        fits = {'test':{'source__'+r:{'valid':False} for r in REPRESENTATIONS}}
        contrasts = {f'source__{r} minus source__raw':{'bce':{'hi95':-.02},'auc':{'delta':.1}}
                     for r in ('particle','relative')}
        content = {'contrasts':{f'{r}_minus_raw/B_minus_A':{'hi95':-.02} for r in ('particle','relative')},
                   'judges':{r:{'B':{'auc':.6}} for r in REPRESENTATIONS}}
        out = experiment.classify_decision(fits,contrasts,content,.002)
        self.assertEqual(out['interpretation'],'unresolved_fit_budget')
        self.assertFalse(any(out['eligible_for_rl'].values()))

    def test_relative_win_needs_raw_control_auc_and_mode_evidence(self):
        fits = {'test':{'source__'+r:{'valid':True} for r in REPRESENTATIONS}}
        contrasts = {f'source__{r} minus source__raw':{'bce':{'hi95':-.02},'auc':{'delta':.1}}
                     for r in ('particle','relative')}
        pair_key = 'source__relative minus source__particle'
        contrasts[pair_key] = {'bce':{'hi95':-.01},'auc':{'delta':.03}}
        content = {'contrasts':{f'{a}_minus_{b}/B_minus_A':{'hi95':-.01}
                    for a,b in [('particle','raw'),('relative','raw'),('relative','particle')]},
                   'judges':{r:{'B':{'auc':v}} for r,v in [('raw',.55),('particle',.6),('relative',.65)]}}
        good = experiment.classify_decision(fits,contrasts,content,.002)
        self.assertEqual(good['interpretation'],'supports_relative_joint_access_advantage')
        self.assertTrue(all(good['eligible_for_rl'].values()))
        # Better than particle alone does not mean better than the raw control.
        contrasts['source__relative minus source__raw']['bce']['hi95'] = .001
        out = experiment.classify_decision(fits,contrasts,content,.002)
        self.assertFalse(out['relative_vs_particle_bce_benefit'])
        self.assertFalse(out['eligible_for_rl']['relative'])
        contrasts['source__relative minus source__raw']['bce']['hi95'] = -.02
        contrasts[pair_key]['auc']['delta'] = -.01
        out = experiment.classify_decision(fits,contrasts,content,.002)
        self.assertFalse(out['relative_vs_particle_bce_benefit'])
        contrasts[pair_key]['auc']['delta'] = .03
        content['contrasts']['relative_minus_particle/B_minus_A']['hi95'] = .001
        out = experiment.classify_decision(fits,contrasts,content,.002)
        self.assertEqual(out['interpretation'],'supports_relative_bce_advantage_content_unresolved')

    def test_classifier_stage_content_rotation_and_plot_plumbing(self):
        """Two-update fixture fits are deliberately INCONCLUSIVE, never science."""
        cfg = self.cfg(); data = Data(cfg); source = native.Denoiser(cfg).eval()
        plan = json.loads((Path(__file__).parent/'plans/relational_classifier.json').read_text())
        plan['bootstrap_repeats'] = 100
        plan['fit'].update(width=8,min_steps=1,max_steps=2,check_every=1)
        pp = {s:panel(data,32,i) for s,i in [('train',1),('validation',2),('test',3)]}
        cc = torch.tensor([[-.75],[-.25],[.25],[.75]])
        yy = data.centers.repeat_interleave(16,0)[None].repeat(4,1,1)
        ids = modes(yy)
        pool = {'c':cc,'y':yy,'ids':ids,'p':data.probabilities(cc[:,0],.9),
                'counts':torch.stack([torch.bincount(row,minlength=8) for row in ids])}
        with tempfile.TemporaryDirectory(dir=lab.ARTIFACTS,prefix='relation_stage1_test_') as tmp:
            root = Path(tmp); old = root/'swaps'; old.mkdir(); output = root/'output'; output.mkdir()
            plan.update(panels=str(root/'panels.pt'),swap_directory=str(old))
            atomic_checkpoint(root/'panels.pt',pp)
            atomic_json(old/'plan.json',{'samples_per_condition':16,'sample_seed':641083})
            atomic_json(old/'report.json',{'state':'completed'})
            atomic_checkpoint(old/'step0_pool.pt',pool)
            with patch.object(lab,'load_source',return_value=(cfg,source)):
                result = experiment.classifier_stage(plan,output,lambda row:None)
            self.assertEqual(result['state'],'inconclusive_fit_budget')
            self.assertTrue(result['paired_original_y_unchanged'])
            self.assertEqual(set(result['content']['judges']),set(REPRESENTATIONS))
            self.assertTrue(all(set(x)=={'A','B','C','D'} for x in result['content']['judges'].values()))
            self.assertEqual(len(result['content']['contrasts']),9)
            self.assertIn('relative_minus_particle/B_minus_A',result['content']['contrasts'])
            self.assertEqual(set(result['rotation']),set(REPRESENTATIONS))
            self.assertFalse(any(result['decision']['eligible_for_rl'].values()))
            atomic_json(output/'plan.json',plan); atomic_json(output/'report.json',result)
            from experiments.dgpo_toy.plot_relational_experiment import plot
            paths = plot(output)
            self.assertEqual(len(paths),1)
            self.assertTrue(Path(paths[0]).is_file())
            self.assertIn('inconclusive_fit_budget',experiment.summarize(plan,result))

    def test_stage2_plumbing_uses_one_reward_and_preserves_milestones(self):
        """Mocks scientific training/audits; tests orchestration, not an experiment."""
        plan = json.loads((Path(__file__).parent/'plans/relational_rl.json').read_text())
        plan.update(eval_contexts=16,eval_candidates=3,structure_grid=4,structure_candidates=16,bootstrap_repeats=100)
        cfg = self.cfg(); source = native.Denoiser(cfg).eval()
        reward = RelationalCritic('relative',width=8).eval().requires_grad_(False)
        provenance = {'test_mock':True,'representation':'relative'}
        seen_rewards,seen_starts = [],[]
        def mock_train(arm,initial,critic,data,current_cfg,seed,monitor_seed,emit,callback,
                       resume_state=None,velocity_coefficient=None):
            self.assertIs(critic,reward); self.assertEqual(velocity_coefficient,1.)
            seen_rewards.append(id(critic)); seen_starts.append(0 if resume_state is None else resume_state['step'])
            current = copy.deepcopy(initial)
            if resume_state is not None:current.load_state_dict(resume_state['model'])
            optimizer = torch.optim.AdamW(current.parameters())
            history = [{'step':i} for i in range(1,current_cfg.policy_steps+1)]
            callback(current_cfg.policy_steps,current,optimizer,native.generator(seed),history)
            return current,history
        def small_panels(model,data,seed):
            return {s:panel(data,16,seed+i,model) for s,i in [('train',1),('validation',2),('test',3)]}
        def mock_fit(panels,reps,output,emit,p):
            lab.assert_pairing(panels)
            scores = {name+'__relative':lab.predictions(reward,pp['test']) for name,pp in panels.items()}
            report = {'test':{k:{**lab.score_metrics(v),'valid':True} for k,v in scores.items()}}
            return report,scores
        with tempfile.TemporaryDirectory(dir=lab.ARTIFACTS,prefix='relation_flow_test_') as tmp, \
             patch.object(experiment,'load_reward',return_value=(reward,provenance)), \
             patch.object(lab,'load_source',return_value=(cfg,source)), \
             patch.object(native,'policy_train',mock_train), \
             patch.object(experiment,'audit_panels',small_panels), \
             patch.object(experiment,'fit',mock_fit):
            out = experiment.rl_stage(plan,Path(tmp),lambda row:None,'relative','Unit-test mock review only')
            self.assertEqual(out['state'],'completed')
            self.assertTrue(out['reward_unchanged'])
            self.assertEqual(seen_starts,[0,0,25,25,100,100,300,300])
            self.assertEqual(len(set(seen_rewards)),1)
            self.assertEqual(set(out['audits']),{'0','25','100','300','1000'})
            self.assertEqual(out['reward_contrasts']['enhanced_minus_raw_1000']['mean'],0)

    def test_wrapper_defaults_local_records_and_stops_after_one_stage(self):
        """Mocked stage execution; no scientific fit and no W&B service."""
        plan = json.loads((Path(__file__).parent/'plans/relational_classifier.json').read_text())
        checks = {'state':'preflight_passed_not_started','training_started':False,
                  'setup':experiment.resolved_setup(plan,self.cfg(),'disabled')}
        result = {'state':'completed','decision':{'test_mock':True}}
        with tempfile.TemporaryDirectory(dir=lab.ARTIFACTS,prefix='relation_wrapper_test_') as tmp, \
             patch.object(experiment,'preflight',return_value=checks), \
             patch.object(experiment,'classifier_stage',return_value=result) as stage, \
             patch.object(experiment,'rl_stage',side_effect=AssertionError('Auto-launched RL')), \
             patch.object(experiment,'summarize',return_value='Mock result, not science'), \
             patch('experiments.dgpo_toy.plot_relational_experiment.plot',return_value=[]), \
             patch.dict('sys.modules',{'wandb':None}):
            output = Path(tmp)/'output'
            experiment.run(plan,output)
            stage.assert_called_once()
            runtime = json.loads((output/'runtime.json').read_text())
            self.assertEqual(runtime['wandb']['mode'],'disabled')
            self.assertIsNone(runtime['policy'])
            self.assertFalse(runtime['next_stage_automatic'])
            self.assertEqual(json.loads((output/'status.json').read_text())['state'],'awaiting_review')
            self.assertFalse(json.loads((output/'preflight.json').read_text())['training_started'])


if __name__ == '__main__':
    unittest.main()
