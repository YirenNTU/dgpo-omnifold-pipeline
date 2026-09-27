import copy
import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

from experiments.dgpo_toy import film_auc_trajectory as runner
from experiments.dgpo_toy import reward_transport_audit as audit
from experiments.dgpo_toy.nonperiodic_cube import Critic, Data
from experiments.dgpo_toy.test_conditioning_transport import config


def make_point(step, scores, valid=True):
    scorer = runner.PairedScores(scores)
    auc, gap, bce = scorer.metrics(np.ones(scorer.n))
    return {'policy_step': step, 'auc': auc, 'auc_gap': gap, 'bce': bce, 'valid': valid,
            'fit_steps': 7500, 'selected_step': 5500, 'plateau': valid}


def test_fast_paired_auc_matches_explicit_bootstrap_including_ties():
    rng = np.random.default_rng(1)
    scores = {k: rng.integers(-3, 4, 87).astype(float) for k in ('positive', 'negative')}
    scorer = runner.PairedScores(scores)
    for _ in range(20):
        ids = rng.integers(87, size=87)
        counts = np.bincount(ids, minlength=87)
        auc, gap, bce = scorer.metrics(counts)
        p, q = scores['positive'][ids], scores['negative'][ids]
        expected = roc_auc_score(np.r_[np.ones(87), np.zeros(87)], np.r_[p, q])
        assert auc == pytest.approx(expected, abs=1e-14)
        assert gap == pytest.approx(abs(expected-.5))
        assert bce == pytest.approx(.5*(np.logaddexp(0., -p)+np.logaddexp(0., q)).mean())


def test_resolved_rise_then_recovery_is_distinct_from_endpoint_failure():
    rng = np.random.default_rng(5)
    base = {k: rng.normal(size=2048) for k in ('positive', 'negative')}
    predictions = {0: {'positive': base['positive']+.6, 'negative': base['negative']},
                   100: {'positive': base['positive']+2., 'negative': base['negative']},
                   3000: {'positive': base['positive'], 'negative': base['negative']}}
    points = [make_point(s, v) for s, v in predictions.items()]
    result = runner.trajectory_analysis(points, predictions, repeats=100)
    assert not result['point_estimate_monotonic']
    assert result['resolved_auc_increase_intervals'] == [[0, 100]]
    assert result['resolved_early_worse_than_start'] == [100]
    assert result['resolved_final_gap_improvement'] and result['resolved_rise_then_recovery']
    assert result['interpretation'] == 'resolved_nonmonotonic_AUC_on_measured_grid'
    assert len(result['contrasts']) == 3
    assert result == runner.trajectory_analysis(points, predictions, repeats=100)


def test_identical_scores_are_not_proof_of_monotonicity():
    score = {'positive': np.linspace(-1, 2, 100), 'negative': np.linspace(-2, 1, 100)}
    predictions = {s: score for s in (0, 100, 500)}
    result = runner.trajectory_analysis([make_point(s, v) for s, v in predictions.items()], predictions, repeats=100)
    assert result['point_estimate_monotonic'] and result['gap_point_estimate_monotonic']
    assert not result['resolved_auc_increase_intervals']
    assert not result['resolved_final_gap_improvement']
    assert result['interpretation'] == 'no_resolved_AUC_increase_not_proof_of_monotonicity'
    assert all(c['auc']['simultaneous_hi95'] == 0 for c in result['contrasts'])


def test_invalid_audit_is_not_bridged_and_bad_panels_are_rejected():
    score = {'positive': np.arange(8.), 'negative': np.arange(8.)}
    points = [make_point(0, score), make_point(100, score, valid=False), make_point(500, score)]
    result = runner.trajectory_analysis(points, {})
    assert result['state'] == 'audit_inconclusive' and result['invalid_steps'] == [100]
    assert result['point_estimate_monotonic'] is None
    for bad in ({'positive': [np.nan, 0], 'negative': [0, 0]},
                {'positive': [1, 2], 'negative': [0]},
                {'positive': [[1, 2]], 'negative': [[1, 2]]}):
        with pytest.raises(ValueError):
            runner.PairedScores(bad)
    points = [make_point(0, score), make_point(100, score)]
    points[-1]['auc'] = .6
    with pytest.raises(ValueError, match='disagree'):
        runner.trajectory_analysis(points, {0: score, 100: score}, repeats=100)


def test_grid_includes_actual_zero_and_is_strictly_increasing():
    assert runner.validate_milestones(runner.MILESTONES) == (0, 100, 250, 500, 750, 1000, 2000, 3000)
    for values in ([], [0], [100, 1000], [0, 0, 100], [0, -1, 100], [0, 100, 50], [False, 100]):
        with pytest.raises(ValueError):
            runner.validate_milestones(values)


def test_single_policy_audit_retains_cold_fit_and_inconclusive_budget(tmp_path, monkeypatch):
    class TinyCritic(torch.nn.Module):
        def __init__(self, fourier):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(.01))

        def forward(self, y, c):
            return self.weight*y[..., 0]

    def panel(data, n, seed, policy):
        return {'c': torch.zeros(n, 1), 'positive': torch.ones(n, 3), 'negative': -torch.ones(n, 3)}

    monkeypatch.setattr(audit, 'Critic', TinyCritic)
    monkeypatch.setattr(audit, 'panel', panel)
    result = audit.fit_audits({'mlp_c': None}, None, tmp_path/'audit', lambda row: None, max_steps=2)
    assert result['cold_start'] and result['seed'] == 41
    assert result['state'] == 'budget_exhausted_inconclusive'
    assert not result['test']['mlp_c']['plateau'] and result['test']['mlp_c']['fit_steps'] == 2
    assert result['contrasts'] == {}
    with pytest.raises(ValueError, match='At least one'):
        audit.fit_audits({}, None, tmp_path/'empty', lambda row: None)


@pytest.mark.parametrize('valid', [True, False])
def test_runner_preserves_native_trajectory_and_separates_wandb_clocks(tmp_path, monkeypatch, valid):
    # Four tiny synthetic policy updates verify transactions; no experiment job.
    cfg = replace(config(), ddim_steps=2, policy_steps=4, eval_events=8, eval_every=2)
    with torch.random.fork_rng():
        torch.manual_seed(23)
        source = runner.film.native.Denoiser(cfg).eval()
        reward = Critic(True, width=8).eval().requires_grad_(False)
    initial = runner.film.make_models(source, cfg, width=32, arms=runner.film.LEARNED_ARMS)['mlp_c']
    before = copy.deepcopy(initial.state_dict())
    monkeypatch.setattr(runner, 'prepare', lambda *a: (cfg, source, initial, {'A': reward}, {'source': 'test-source'}, {}))
    monkeypatch.setattr(runner.film.transport, 'assess', lambda *a, **kw: ({'mode_tv': .2}, {}))
    reward_dir = tmp_path/'reward'; reward_dir.mkdir()
    torch.save({'model': reward.state_dict(), 'selected_step': 3100}, reward_dir/'classifier.pt')
    fits, calls = [], []
    score = {'positive': torch.linspace(-1, 2, 32), 'negative': torch.linspace(-2, 1, 32)}

    def fake_audit(policies, data, output, emit, max_steps):
        assert list(policies) == ['mlp_c'] and max_steps == 16000
        fits.append(str(output))
        output.mkdir()
        shared = {'c': torch.zeros(8, 1), 'positive': torch.ones(8, 3), 'negative': torch.zeros(8, 3)}
        torch.save(dict.fromkeys(('train', 'validation', 'test'), shared), output/'mlp_c_panels.pt')
        torch.save({'mlp_c': score}, output/'test_scores.pt')
        emit({'phase': 'audit', 'step': 7500, 'validation_auc': .7})
        stats = make_point(0, score, valid=valid)
        torch.manual_seed(555)  # Audit must not perturb the private policy RNG.
        return {'state': 'completed' if valid else 'budget_exhausted_inconclusive', 'test': {'mlp_c': stats}}

    native_train = runner.film.native.policy_train
    full_states = []

    def capture(s, m, opt, rng, h):
        full_states.append(copy.deepcopy({'model': m.state_dict(), 'optimizer': opt.state_dict(), 'rng': rng.get_state()}))

    native_train('dgpo', initial, reward, Data(cfg), cfg, 17, 284017, lambda row: None,
                 capture, velocity_coefficient=1.)

    def traced_train(*args, **kwargs):
        saved = kwargs['resume_state']
        calls.append((0 if saved is None else saved['step'], args[4].policy_steps))
        assert args[1] is initial and args[2] is reward and kwargs['velocity_coefficient'] == 1.
        return native_train(*args, **kwargs)

    metrics, logged, metadata, finished = [], [], {}, []
    wb = SimpleNamespace(id='testid', dir='test-files', summary={},
                         define_metric=lambda *a, **kw: metrics.append((a, kw)),
                         log=lambda row: logged.append(row), finish=lambda **kw: finished.append(kw))

    def init(**kwargs):
        metadata.update(kwargs)
        return wb

    monkeypatch.setattr(runner, 'fit_audits', fake_audit)
    monkeypatch.setattr(runner.film.native, 'policy_train', traced_train)
    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(init=init))
    output = tmp_path/'out'
    result = runner.run(reward_dir, tmp_path/'control', output, milestones=(0, 2, 4),
                        bootstrap_repeats=100, wandb_mode='offline', run_id='testid')
    assert calls == [(0, 2), (2, 4)] and len(fits) == 3
    assert [p['policy_step'] for p in result['points']] == [0, 2, 4]
    assert result['state'] == ('completed' if valid else 'audit_inconclusive')
    assert all(result['fixed_states_unchanged'].values())
    assert runner.film.previous.equal_state(initial.state_dict(), before)
    state = torch.load(output/'mlp_c_last.pt', weights_only=True)
    full = full_states[-1]
    assert runner.film.previous.equal_state(state['model'], full['model'])
    assert torch.equal(state['rng'], full['rng'])
    for key, value in full['optimizer']['state'].items():
        for field, tensor in value.items():
            assert torch.equal(tensor, state['optimizer']['state'][key][field])
    assert metadata['name'] == runner.RUN_NAME and metadata['id'] == 'testid'
    assert metadata['config']['reward_arm'] == 'A'
    assert (('validation/*',), {'step_metric': 'validation/step'}) in metrics
    assert (('audit/policy2/*',), {'step_metric': 'audit/policy2/step'}) in metrics
    assert any(row.get('validation/step') == 2 and row['validation/fit_steps'] == 7500 for row in logged)
    assert any(row.get('audit/policy2/step') == 7500 for row in logged)
    assert finished == [{'exit_code': 0}]
    assert (output/'auc_trajectory.png').is_file() and (output/'monotonicity.json').is_file()
    assert (output/'auc_trajectory.csv').read_text().count('\n') == 4


def test_panel_pairing_checks_truth_and_context(tmp_path):
    shared = {'c': torch.zeros(4, 1), 'positive': torch.ones(4, 3)}
    panels = {s: copy.deepcopy(shared) for s in ('train', 'validation', 'test')}
    a, b = tmp_path/'a.pt', tmp_path/'b.pt'
    torch.save(panels, a); torch.save(panels, b)
    runner.verify_panels(a, b)
    panels['test']['positive'][0, 0] = 9
    torch.save(panels, b)
    with pytest.raises(ValueError, match='identities changed'):
        runner.verify_panels(a, b)


def test_preflight_does_not_train_or_create_output(tmp_path, monkeypatch, capsys):
    cfg = config()
    reward = Critic(True, width=8).requires_grad_(False)
    monkeypatch.setattr(runner, 'prepare', lambda *a: (cfg, None, None, {'A': reward}, {'source': 'fixture'}, {}))
    monkeypatch.setattr(runner, 'run', lambda *a, **kw: pytest.fail('No training in preflight'))
    output = tmp_path/'not-created'
    monkeypatch.setattr(sys, 'argv', ['trajectory', '--preflight', '--output', str(output)])
    runner.main()
    result = json.loads(capsys.readouterr().out)
    assert result['state'] == 'preflight_passed_not_started' and result['reward_frozen']
    assert not output.exists()
