import copy
import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from experiments.dgpo_toy import reward_shape_failure as runner
from experiments.dgpo_toy.test_conditioning_transport import config


def test_dense_audit_default_preserves_final_endpoint_and_shared_baseline():
    assert runner.MILESTONES == (25, 50, 100, 150, 200, 300, 1000)
    assert runner.run.__kwdefaults__['milestones'] == runner.MILESTONES
    assert runner.film.validate_milestones(runner.MILESTONES) == runner.MILESTONES
    assert runner.MILESTONES[-1] == 1000
    assert 1+len(runner.ARMS)*len(runner.MILESTONES) == 15


def test_paired_reward_targets_share_exact_modes_contexts_and_negatives():
    c = torch.tensor([[-.5], [.5]])
    full = torch.tensor([[1., -1., 1.], [-1., 1., -1.]])
    shape, negative = full*.9, full.flip(0)
    swap = {key: {'c': c, 'negative': value} for key, value in
            [('A', full), ('C', shape), ('D', negative)]}
    p = runner.paired_targets(swap)
    assert p['full']['negative'] is p['shape']['negative']
    assert p['full']['c'] is p['shape']['c']
    assert torch.equal(runner.modes(p['full']['positive']), runner.modes(p['shape']['positive']))
    assert not torch.equal(p['full']['positive'], p['shape']['positive'])
    swap['C']['negative'] = -shape
    with pytest.raises(ValueError, match='mode identities'):
        runner.paired_targets(swap)


def test_same_targets_replay_identical_reward_fit_and_short_budget_is_invalid(tmp_path):
    rng = runner.native.generator(5)
    p = {'c': torch.zeros(512, 1), 'positive': torch.randn(512, 3, generator=rng),
         'negative': torch.randn(512, 3, generator=rng)}
    panels = {split: {arm: p for arm in runner.ARMS} for split in ('train', 'validation', 'test')}
    models, result = runner.fit_rewards(panels, tmp_path, lambda row: None, max_steps=2)
    assert runner.film.previous.equal_state(models['full'].state_dict(), models['shape'].state_dict())
    assert not result['valid'] and result['state'] == 'inconclusive_reward_fit'
    assert all(not p.requires_grad for m in models.values() for p in m.parameters())
    assert result['arms']['full'] == result['arms']['shape']


def contrast(gap, bce):
    return {'auc_gap': {'lo95': gap[0], 'hi95': gap[1]},
            'bce': {'lo95': bce[0], 'hi95': bce[1]}}


def test_failure_requires_resolved_reward_gain_and_excludes_material_closure():
    contrasts = {'full_minus_baseline': contrast((.001, .02), (-.02, .001)),
                 'shape_minus_baseline': contrast((-.06, -.03), (.005, .025)),
                 'full_minus_shape': contrast((.02, .09), (-.03, -.01))}
    gains = {'full': {'lo95': .02}}
    result = runner.decide(True, contrasts, gains)
    assert result['failure_reproduced'] and result['shape_control_rescue']
    assert result['interpretation'] == 'supports_reward_target_package_as_contributor'
    # Wide confidence interval is inconclusive, not a reproduced plateau.
    contrasts['full_minus_baseline'] = contrast((-.04, .04), (-.04, .04))
    assert not runner.decide(True, contrasts, gains)['failure_reproduced']
    assert runner.decide(False, contrasts, gains)['failure_reproduced'] is None
    contrasts['full_minus_baseline'] = contrast((.001, .02), (-.02, .001))
    gains['full']['lo95'] = -.01
    assert not runner.decide(True, contrasts, gains)['failure_reproduced']


def test_simultaneous_audit_intervals_and_paired_identity():
    rng = np.random.default_rng(7)
    p, q = rng.normal(size=(2, 1024))
    scores = {name: {'positive': p+delta, 'negative': q}
              for name, delta in [('baseline', .7), ('full', 1.2), ('shape', .1)]}
    result = runner.paired_audit_contrasts(scores, repeats=100)
    assert result['full_minus_baseline']['auc_gap']['lo95'] > 0
    assert result['shape_minus_baseline']['auc_gap']['hi95'] < 0
    assert result['full_minus_shape']['auc_gap']['lo95'] > 0
    identical = runner.paired_audit_contrasts(dict.fromkeys(scores, scores['baseline']), repeats=100)
    assert all(v['auc_gap']['lo95'] == v['auc_gap']['hi95'] == 0 for v in identical.values())


def test_reward_decomposition_adds_exactly_and_reports_missing_support():
    before = {'contexts': torch.zeros(2, 1), 'q': torch.full((2, 8), .125),
              'cells': {'full': {'mean': torch.arange(8.).expand(2, -1),
                                 'mean_available': torch.ones(2, 8, dtype=torch.bool)}},
              'rewards': {'full': torch.zeros(2, 8)}}
    after = copy.deepcopy(before)
    after['q'][:, 0] -= .05
    after['q'][:, 7] += .05
    after['rewards']['full'] += .5
    d = runner.decompose(after, before, 'full')
    assert d['total_gain'] == pytest.approx(d['mode_probability']+d['within_mode_shape_and_interaction'])
    assert d['mode_probability'] == pytest.approx(.35)
    before['cells']['full']['mean_available'][0, 0] = False
    assert not runner.decompose(after, before, 'full')['available']


@pytest.mark.parametrize('milestones', [(2, 4), (1, 2, 3, 4)])
def test_runner_uses_same_initial_native_reference_and_keeps_optimizer_rng(tmp_path, monkeypatch, milestones):
    cfg = replace(config(), ddim_steps=2, policy_steps=4, eval_events=8, eval_every=2)
    with torch.random.fork_rng():
        torch.manual_seed(23)
        source = runner.native.Denoiser(cfg).eval()
        initial = runner.film.NonlinearFiLM(cfg, source.state_dict(), width=4).eval()
        reward = runner.Critic(True, width=8).eval().requires_grad_(False)
    rewards = {arm: copy.deepcopy(reward) for arm in runner.ARMS}
    initial_state = copy.deepcopy(initial.state_dict())
    monkeypatch.setattr(runner, 'prepare', lambda *a: (cfg, source, initial, {}))
    monkeypatch.setattr(runner, 'make_reward_panels', lambda *a: ({}, {}))
    monkeypatch.setattr(runner, 'fit_rewards', lambda *a: (rewards, {'valid': True}))
    endpoint = {'contexts': torch.zeros(4, 1), 'q': torch.full((4, 8), .125),
        'rewards': {name: torch.zeros(4, 8) for name in rewards},
        'cells': {name: {'mean': torch.zeros(4, 8), 'mean_available': torch.ones(4, 8, dtype=torch.bool)}
                  for name in rewards}}
    monkeypatch.setattr(runner, 'measure', lambda *a: ({'mode_tv': .1}, endpoint))
    fits, verified = [], []

    def fake_audit(policies, data, folder, emit, max_steps):
        name = next(iter(policies))
        fits.append(name)
        # Real cold fits use global initialization RNG. Extra fits must not
        # perturb the saved policy RNG/AdamW trajectory at any cadence.
        torch.manual_seed(41)
        torch.rand(31)
        folder.mkdir(parents=True)
        torch.save({name: {'positive': torch.linspace(-1, 2, 128),
                          'negative': torch.linspace(-2, 1, 128)}}, folder/'test_scores.pt')
        return {'state': 'completed', 'test': {name: {'auc': .7, 'bce': .6, 'auc_gap': .2,
                'fit_steps': 3000, 'selected_step': 1000, 'plateau': True}}}

    monkeypatch.setattr(runner, 'fit_audits', fake_audit)
    monkeypatch.setattr(runner, 'verify_panels', lambda a, b: verified.append((a, b)))
    output = tmp_path/'run'
    result = runner.run(tmp_path/'source.pt', output, milestones=milestones,
                        bootstrap_repeats=100, wandb_mode='disabled')
    assert result['state'] == 'completed' and result['fixed_states_unchanged']
    assert fits == ['baseline']+list(runner.ARMS)*len(milestones)
    assert len(verified) == 2*len(milestones)
    assert result['audit']['policy_steps'] == [0, *milestones]
    assert result['audit']['planned_cold_fits'] == len(fits)
    assert result['primary_endpoint'] == result['config']['policy_steps'] == milestones[-1]
    rows = [json.loads(line) for line in (output/'progress.jsonl').read_text().splitlines()]
    for arm in runner.ARMS:
        validation = [row for row in rows if row['phase'] == 'validation' and row['arm'] == arm]
        assert [row['step'] for row in validation] == [0, *milestones]
        assert validation[0]['shared_baseline'] is True
        assert validation[0]['fit_steps'] == result['baseline_audit']['fit_steps']
    assert runner.film.previous.equal_state(initial.state_dict(), initial_state)
    native_states = []
    runner.native.policy_train('dgpo', initial, reward, runner.Data(cfg), cfg, 17, 284017,
        lambda row: None, lambda step, m, opt, rng, history:
        native_states.append({'model': copy.deepcopy(m.state_dict()), 'rng': rng.get_state(),
                              'optimizer': copy.deepcopy(opt.state_dict())}), velocity_coefficient=1.)
    for name in runner.ARMS:
        saved = torch.load(output/f'{name}_last.pt', weights_only=True)
        assert runner.film.previous.equal_state(saved['model'], native_states[-1]['model'])
        assert torch.equal(saved['rng'], native_states[-1]['rng'])
        assert saved['velocity_coefficient'] == 1 and saved['reward_arm'] == name
        assert saved['step'] == len(saved['history']) == 4
        for parameter, state in saved['optimizer']['state'].items():
            for key, value in state.items():
                assert torch.equal(value, native_states[-1]['optimizer']['state'][parameter][key])
    assert json.loads((output/'report.json').read_text())['state'] == 'completed'
