import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.dgpo_toy import closed_loop_lab as lab
from experiments.dgpo_toy import conditional as native
from experiments.dgpo_toy.nonperiodic_cube import Critic, Data, panel
from experiments.dgpo_toy.film_conditioning import NonlinearFiLM


class ClosedLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_fourier_masks_and_original_equivalence(self):
        y, c = torch.randn(16, 3), torch.randn(16, 1)
        base = Critic(True)
        both = lab.FeatureCritic('both')
        both.load_state_dict(base.state_dict())
        self.assertTrue(torch.equal(base(y, c), both(y, c)))
        parts = {k: lab.FeatureCritic(k).features(y, c) for k in lab.FEATURES}
        self.assertTrue(torch.equal(parts['plain'][:, :4], parts['both'][:, :4]))
        self.assertEqual(float(parts['plain'][:, 4:].abs().sum()), 0)
        self.assertTrue(torch.equal(parts['c_only'][:, 4:]+parts['y_only'][:, 4:], parts['both'][:, 4:]))
        self.assertTrue(torch.equal(parts['c_only'][:, 4:].reshape(16, 8, 4)[:, :, 1:], torch.zeros(16, 8, 3)))

    def config(self, steps=4):
        return native.Config(dimensions=3, context_dim=1, hidden=8, ddim_steps=4,
            policy_steps=steps, batch=4, candidates=3, timesteps=2, eval_events=8, eval_every=2)

    def test_all_initial_functions_and_parameter_counts_match(self):
        cfg = self.config()
        source = native.Denoiser(cfg)
        x, c, t = torch.randn(8, 3), torch.randn(8, 1), torch.rand(8)
        counts = []
        for spec in ({}, {'basis': 'raw'}, {'basis': 'polynomial'}, {'modulation': 'shift'},
                     {'modulation': 'scale'}, {'nonlinear': False}, {'normalization': False}, {'layers': 1}):
            model = lab.ConditioningAblation(cfg, source.state_dict(), **spec)
            self.assertTrue(torch.equal(source(x, t, c), model(x, t, c)))
            self.assertTrue(torch.equal(native.ddim(source, c, x, 4), native.ddim(model, c, x, 4)))
            counts.append(sum(p.numel() for p in model.parameters()))
        self.assertEqual(len(set(counts)), 1)

    def test_full_arm_matches_prior_nonlinear_film(self):
        cfg = self.config()
        source = native.Denoiser(cfg)
        previous = NonlinearFiLM(cfg, source.state_dict())
        current = lab.ConditioningAblation(cfg, source.state_dict())
        with torch.no_grad():
            for head in previous.condition_heads:
                head.weight.normal_(std=.01)
        current.load_state_dict(previous.state_dict())
        x, c, t = torch.randn(8, 3), torch.randn(8, 1), torch.rand(8)
        self.assertTrue(torch.equal(current(x, t, c), previous(x, t, c)))

    def test_no_global_rng_consumption_and_exact_resume(self):
        cfg = self.config()
        source = native.Denoiser(cfg)
        model = lab.ConditioningAblation(cfg, source.state_dict())
        critic = lab.FeatureCritic().requires_grad_(False)
        data, saved = Data(cfg), {}
        def checkpoint(step, m, opt, rng, hist):
            saved.update(model=copy.deepcopy(m.state_dict()), optimizer=copy.deepcopy(opt.state_dict()),
                         rng=rng.get_state(), step=step, history=copy.deepcopy(hist), velocity_coefficient=1.)
        before = torch.get_rng_state().clone()
        full, _ = native.policy_train('dgpo', model, critic, data, cfg, 27, 70, lambda _: None,
                                      velocity_coefficient=1.)
        short_cfg = lab.replace(cfg, policy_steps=2)
        native.policy_train('dgpo', model, critic, data, short_cfg, 27, 70, lambda _: None,
                            checkpoint, velocity_coefficient=1.)
        resumed, _ = native.policy_train('dgpo', model, critic, data, cfg, 27, 70, lambda _: None,
                                        resume_state=saved, velocity_coefficient=1.)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        for key, value in full.state_dict().items():
            self.assertTrue(torch.equal(value, resumed.state_dict()[key]), key)

    def test_scope_and_round_cap(self):
        with self.assertRaises(ValueError):
            lab.toy_path('/tmp/not_toy')
        plan = {'round': 21, 'kind': 'classifier', 'question': 'q', 'primary_endpoint': 'bce',
                'decision_rule': 'd', 'classifier_seed': 1, 'panel_seed': 2, 'max_fit_steps': 32000,
                'bootstrap_repeats': 100, 'run_name': 'Question | toy | paired'}
        with self.assertRaises(ValueError):
            lab.validate_plan(plan)

    def test_paired_intervals(self):
        scores = {'a': {'positive': np.array([2., 1., 0.]), 'negative': np.array([0., -1., 0.])}}
        scores['b'] = scores['a']
        out = lab.compare_scores(scores, [('a', 'b')], repeats=100)
        self.assertTrue(all(m == {'delta': 0., 'lo95': 0., 'hi95': 0.} for m in out['a minus b'].values()))

    def test_tiny_fit_is_explicitly_inconclusive(self):
        data = Data(self.config())
        panels = {'source': {k: panel(data, 32, seed) for k, seed in
                            [('train', 11), ('validation', 12), ('test', 13)]}}
        lab.ARTIFACTS.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='closed_loop_test_', dir=lab.ARTIFACTS) as tmp:
            result, _ = lab.fit_classifiers(panels, ['plain', 'both'], Path(tmp)/'fits', lambda _: None,
                    max_steps=2, min_steps=1, patience=20, check_every=1)
            self.assertEqual(result['state'], 'inconclusive_fit_budget')
            self.assertTrue(all(not s['valid'] for s in result['test'].values()))
            for name, stats in result['test'].items():
                expected = min((r for r in result['history'] if r['arm'] == name), key=lambda r: r['validation_bce'])
                self.assertEqual(expected['step'], stats['selected_step'])

    def test_rl_round_wiring_with_tiny_mock_audits(self):
        """Exercise stage/resume, panel pairing and contrast names, not scientific adequacy."""
        cfg = self.config()
        source = native.Denoiser(cfg).eval()
        plan = {'conditioning': {'fourier': {}, 'raw': {'basis': 'raw'}},
                'initialization_seed': 17, 'policy_seed': 27, 'monitor_seed': 37,
                'panel_seed': 47, 'classifier_seed': 57, 'max_fit_steps': 2000,
                'milestones': [2, 4], 'bootstrap_repeats': 100}
        def small_panels(policy, data, seed):
            return {split: panel(data, 32, seed+offset, policy) for split, offset in
                    [('train', 1), ('validation', 2), ('test', 3)]}
        def mock_audit(panels, modes, output, emit, **kwargs):
            lab.assert_pairing(panels)
            classifier = lab.FeatureCritic()
            ss = {f'{name}__both': lab.predictions(classifier, p['test']) for name, p in panels.items()}
            stats = {name: {**lab.score_metrics(value), 'valid': True} for name, value in ss.items()}
            return {'state': 'completed', 'test': stats, 'unit_test_mock_only': True}, ss
        original_measure = lab.previous.measure
        def small_measure(model, rewards, data, **kwargs):
            return original_measure(model, rewards, data, grid=4, k=16)
        with tempfile.TemporaryDirectory(prefix='closed_loop_wiring_test_', dir=lab.ARTIFACTS) as tmp:
            with patch.object(lab, 'load_source', return_value=(cfg, source)), \
                 patch.object(lab, 'generate_panels', small_panels), \
                 patch.object(lab, 'fit_classifiers', mock_audit), \
                 patch.object(lab.previous, 'measure', small_measure):
                result = lab.rl_round(plan, Path(tmp), lambda _: None)
            self.assertEqual(result['state'], 'completed')
            self.assertEqual(set(result['audits']), {'0', '2', '4'})
            self.assertIn('raw__both_step4 minus fourier__both_step4', result['contrasts'])
            self.assertIn('fourier__both_step4 minus baseline__both_step0', result['contrasts'])
            state = torch.load(Path(tmp)/'fourier_last.pt', weights_only=True)
            self.assertEqual(state['step'], 4)
            self.assertEqual(len(state['history']), 4)


if __name__ == '__main__':
    unittest.main()
