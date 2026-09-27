import copy
import json
from dataclasses import asdict, replace

import pytest
import torch

from experiments.dgpo_toy import conditional as native
from experiments.dgpo_toy import reward_transport as runner
from experiments.dgpo_toy import reward_transport_audit as audit
from experiments.dgpo_toy.cube_lockdown import ConditionDenoiser
from experiments.dgpo_toy.nonperiodic_cube import Critic, Data


def small_config():
    return native.Config(dimensions=3, context_dim=1, hidden=8, ddim_steps=50,
                         policy_steps=2, eval_every=100, eval_events=512,
                         batch=4, candidates=3, timesteps=2)


def test_interpolation_boundaries_and_arbitrary_shapes():
    nodes = torch.tensor([-1., 0., 1.])
    table = torch.stack([torch.arange(8.) + v for v in nodes])
    c = torch.tensor([[-1., -.25], [0., 1.]])
    got = runner.interpolate(nodes, table, c)
    assert torch.allclose(got, c[..., None] + torch.arange(8.))
    with pytest.raises(ValueError, match='outside'):
        runner.interpolate(nodes, table, torch.tensor([1.1]))
    with pytest.raises(ValueError):
        runner.ModeTableReward(nodes.flip(0), table)


def test_B_invariant_within_mode_and_preserves_known_conditional_mean():
    nodes = torch.linspace(-1, 1, 3)
    means = torch.stack([torch.arange(8.) + c for c in nodes])
    reward = runner.ModeTableReward(nodes, means)
    data = Data(small_config())
    c = torch.tensor([[-.4], [.7]])
    y = data.centers[None].expand(2, -1, -1)
    a = reward(y * .4, c[:, None])
    b = reward(y * 2., c[:, None])
    assert torch.equal(a, b)
    assert torch.allclose(a, torch.arange(8.) + c)


def test_C_uses_actual_nonuniform_denominator_and_truth_numerator():
    data = Data(small_config())
    q = torch.tensor([.05, .10, .15, .20, .05, .10, .15, .20])
    reward = runner.ModeTableReward(torch.tensor([-1., 1.]), q.repeat(2, 1), oracle=True)
    c = torch.tensor([[-.77], [.3]])
    y = data.centers[None].expand(2, -1, -1)
    scores = reward(y, c[:, None], data)
    p = data.probabilities(c[:, 0], .9)
    assert torch.allclose(q * scores.exp(), p, atol=1e-7)
    assert not torch.allclose(scores, (p / .125).log())
    assert torch.equal(scores, reward(y * .7, c[:, None], data))
    with pytest.raises(ValueError):
        runner.ModeTableReward(torch.tensor([-1., 1.]), torch.ones(2, 8), oracle=True)


def test_collapse_is_measured_without_inventing_empty_cell_means():
    data = Data(small_config())
    c = torch.tensor([[-.5], [.5]])
    y = torch.full((2, 128, 3), 2.)
    stats, payload = runner.sample_metrics(y, c, data)
    assert stats['missing_mode_cells'] == 14
    assert stats['corner_fraction'] == 0
    assert stats['mode_tv'] > .7
    assert torch.equal(payload['q'].sum(-1), torch.ones(2, dtype=torch.float64))
    cell = runner.cells(y, torch.ones(2, 128))
    assert int(cell['mean_available'].sum()) == 2
    assert torch.isfinite(cell['mean']).all()


def test_decomposition_works_when_endpoint_loses_modes():
    data = Data(small_config())
    y = data.centers[None].repeat(2, 4, 1)
    r = torch.arange(8.).repeat(2, 4)
    base_cells = runner.cells(y, r)
    base = {'q': base_cells['q'], 'learned_cells': base_cells, 'rewards': {'A': r}}
    current = {'q': torch.nn.functional.one_hot(torch.tensor([7, 7]), 8).double(),
               'rewards': {'A': torch.full_like(r, 7.)}}
    result = runner.learned_decomposition(current, base)
    assert result['available'] and result['total'] == result['mode_probability'] == 3.5
    assert result['within_mode_shape'] == 0


def test_table_reward_works_with_unchanged_native_dgpo_gradient():
    cfg = replace(small_config(), ddim_steps=2)
    data = Data(cfg)
    initial = native.Denoiser(cfg)
    model = ConditionDenoiser(cfg, 'fourier', initial.state_dict())
    reference = copy.deepcopy(model).requires_grad_(False)
    critic = runner.ModeTableReward(torch.tensor([-1., 1.]), torch.arange(8.).repeat(2, 1))
    rng = native.generator(3)
    c = data.contexts(cfg.batch, rng)
    noise = torch.randn(cfg.batch, cfg.candidates, 3, generator=rng)
    t = .7 * torch.rand(cfg.timesteps, cfg.batch, generator=rng)
    eps = torch.randn(cfg.timesteps, cfg.batch, 3, generator=rng)
    loss, stats = native.dgpo_objective(model, reference, critic, data, cfg, c, noise, t, eps,
                                         velocity_coefficient=1.)
    loss.backward()
    assert stats['velocity_coefficient'] == 1
    assert stats['velocity_mse'] == 0  # exact matched start
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert sum(float(p.grad.square().sum()) for p in model.parameters()) > 0
    assert not any(p.requires_grad for p in critic.parameters())


def test_calibration_uses_independent_contexts_and_seeds(monkeypatch):
    cfg = small_config()
    data = Data(cfg)
    calls = []

    def generate(model, c, k, seed, steps=50):
        calls.append((c.clone(), seed))
        return data.centers.repeat(k // 8, 1)[None].repeat(len(c), 1, 1)

    class Ideal(torch.nn.Module):
        def forward(self, y, c, ignored=None):
            p = data.probabilities(c.expand(*y.shape[:-1], 1)[..., 0], .9)
            return (p / .125).log().gather(-1, runner.modes(y)[..., None]).squeeze(-1)

    monkeypatch.setattr(runner, 'generate', generate)
    rewards, tensors, report = runner.calibrate(None, Ideal(), data, nodes=257, draws=1024, validation_draws=512)
    assert report['passed']
    assert calls[0][1] != calls[1][1]
    assert not torch.isin(calls[1][0], calls[0][0]).any()
    assert torch.equal(tensors['q_initial'], torch.full((257, 8), .125, dtype=torch.float64))
    assert report['C_reweighting_mode_tv_after'] < 1e-7
    assert set(rewards) == set(runner.ARMS)


def test_calibration_sparse_source_stops_without_fabricating_scores(monkeypatch):
    monkeypatch.setattr(runner, 'generate', lambda model, c, k, seed: torch.ones(len(c), k, 3))
    with pytest.raises(ValueError, match='Calibration insufficient'):
        runner.calibrate(None, Critic(True), Data(small_config()), nodes=3, draws=64)


def test_audit_paired_contrasts_include_bce_and_gap():
    baseline = {'positive': torch.ones(12), 'negative': torch.zeros(12)}
    chance = {'positive': torch.zeros(12), 'negative': torch.zeros(12)}
    results = audit.audit_contrasts({'baseline': baseline, 'A': baseline, 'B': chance, 'C': chance}, repeats=5)
    assert results['A_minus_baseline']['auc']['delta'] == 0
    assert results['B_minus_A']['auc_gap']['hi95'] == -.5
    assert results['B_minus_baseline']['bce']['lo95'] > 0
    assert results['C_minus_B']['bce']['delta'] == 0


def test_decision_never_calls_missing_or_undertrained_audit_closure():
    report = {'baseline': {'mode_tv': .2, 'corner_fraction': .95}, 'arms': {}}
    for name, tv in zip(runner.ARMS, [.2, .1, .1]):
        report['arms'][name] = {'endpoint': {'mode_tv': tv, 'corner_fraction': .95, 'missing_mode_cells': 0}}
    result = runner.decisions(report)
    assert result['interpretation'].startswith('supports_within_mode')
    assert not result['fresh_audit_valid']
    report['audit'] = {'state': 'budget_exhausted_inconclusive'}
    assert not any(runner.decisions(report)['fresh_auc_gap_improves'].values())


def write_source(tmp_path):
    source, previous = tmp_path / 'source', tmp_path / 'previous'
    source.mkdir()
    previous.mkdir()
    cfg = small_config()
    torch.save({'model': native.Denoiser(cfg).state_dict(), 'config': asdict(cfg),
                'trained_on': 'uniform cube reference, not full truth'}, source / 'reference.pt')
    ck = {'model': Critic(True).state_dict(), 'selected_step': 3100}
    torch.save(ck, previous / 'classifier.pt')
    prior = {'state': 'completed', 'source': str(source.resolve()), 'config': asdict(cfg),
             'velocity_coefficient': 1.,
             'classifier': {'gate': {'passed': True}, 'selected_step': 3100}}
    (previous / 'report.json').write_text(json.dumps(prior))
    return source, previous


def test_preflight_validates_source_and_selected_classifier(tmp_path):
    source, previous = write_source(tmp_path)
    _, _, critic, ck, _ = runner.load_source(source, previous, 2)
    assert ck['selected_step'] == 3100
    assert not any(p.requires_grad for p in critic.parameters())
    bad = torch.load(previous / 'classifier.pt', weights_only=True)
    bad['selected_step'] = 3200
    torch.save(bad, previous / 'classifier.pt')
    with pytest.raises(ValueError, match='selection mismatch'):
        runner.load_source(source, previous, 2)


def test_runner_transactions_and_independent_arm_state_without_training(tmp_path, monkeypatch):
    source, previous = write_source(tmp_path)
    # Do not claim a matched old endpoint at this mocked transaction budget.
    meta = json.loads((previous / 'report.json').read_text())
    meta['config']['policy_steps'] = 1000
    (previous / 'report.json').write_text(json.dumps(meta))
    cfg, initial, _, _, _ = runner.load_source(source, previous, 2)
    calls = []

    def calibrate(model, critic, data):
        means = torch.arange(8.).repeat(2, 1)
        return {'A': critic, 'B': runner.ModeTableReward(torch.tensor([-1., 1.]), means),
                'C': runner.ModeTableReward(torch.tensor([-1., 1.]), torch.full((2, 8), .125), oracle=True)}, {}, {'passed': True}

    def assess(model, rewards, data, seed, **kwargs):
        c = torch.tensor([[-.5], [.5]])
        y = data.centers[None].repeat(2, 2, 1)
        stats, payload = runner.sample_metrics(y, c, data)
        payload['rewards'] = {k: r(y, c[:, None], data) for k, r in rewards.items()}
        payload['learned_cells'] = runner.cells(y, payload['rewards']['A'])
        return stats, payload

    def policy_train(arm, model, critic, data, actual_cfg, seed, monitor_seed, emit, callback, **kwargs):
        assert kwargs['velocity_coefficient'] == 1.
        calls.append((seed, monitor_seed, copy.deepcopy(model.state_dict())))
        assert all(torch.equal(v, initial.state_dict()[k]) for k, v in model.state_dict().items()
                   if not k.startswith('condition_adapter'))
        returned = copy.deepcopy(model)
        opt = torch.optim.AdamW(returned.parameters())
        callback(2, returned, opt, native.generator(seed), [{'step': 1}, {'step': 2}])
        return returned, [{'step': 1}, {'step': 2}]

    monkeypatch.setattr(runner, 'calibrate', calibrate)
    monkeypatch.setattr(runner, 'assess', assess)
    monkeypatch.setattr(runner, 'verify_initial', lambda *args: {'mock': True})
    monkeypatch.setattr(native, 'policy_train', policy_train)
    report = runner.run(source, previous, tmp_path / 'out', steps=2, wandb_mode='disabled', skip_audit=True)
    assert report['state'] == 'completed_without_audit'
    assert report['source_unchanged'] and report['classifier_unchanged']
    assert all(report['rewards_unchanged'].values())
    assert len(calls) == 3 and len({(c[0], c[1]) for c in calls}) == 1
    assert all(torch.equal(calls[0][2][k], call[2][k]) for call in calls for k in call[2])
    assert all((tmp_path / 'out' / f'{arm}_last.pt').is_file() for arm in runner.ARMS)
    assert report['decision']['fresh_auc_gap_improves'] == {}


def test_wandb_name_and_upload_exclusions():
    from pathlib import Path
    assert len(runner.RUN_NAME) <= 96 and '_' not in runner.RUN_NAME
    excluded = Path('NERSC/upload-excludes.txt').read_text()
    assert '/experiments/dgpo_toy/' in excluded and '/artifacts/dgpo_toy/' in excluded


def test_failed_calibration_stops_before_any_policy_or_audit(tmp_path, monkeypatch):
    source, previous = write_source(tmp_path)
    monkeypatch.setattr(runner, 'verify_initial', lambda *args: {})
    monkeypatch.setattr(runner, 'calibrate', lambda *args: ({}, {}, {'passed': False}))
    def forbidden(*args, **kwargs):
        raise AssertionError('A failed validity gate must stop before training')
    monkeypatch.setattr(native, 'policy_train', forbidden)
    monkeypatch.setattr(runner, 'fit_audits', forbidden)
    report = runner.run(source, previous, tmp_path / 'out', steps=2, wandb_mode='disabled')
    assert report['state'] == 'stopped_calibration_gate'
    assert report['arms'] == {}
    assert json.loads((tmp_path / 'out' / 'report.json').read_text())['state'] == report['state']


def test_sparse_calibration_error_is_persisted(tmp_path, monkeypatch):
    source, previous = write_source(tmp_path)
    monkeypatch.setattr(runner, 'verify_initial', lambda *args: {})
    def broken(*args):
        raise ValueError('Missing calibration support')
    monkeypatch.setattr(runner, 'calibrate', broken)
    with pytest.raises(ValueError, match='Missing calibration'):
        runner.run(source, previous, tmp_path / 'out', steps=2, wandb_mode='disabled')
    report = json.loads((tmp_path / 'out' / 'report.json').read_text())
    assert report['state'] == 'failed_or_interrupted'
    assert 'Missing calibration' in report['error']


def test_audit_pipeline_cold_start_pairing_and_budget_exhaustion(tmp_path, monkeypatch):
    # Tiny synthetic plumbing test, NOT training an experimental classifier.
    class TinyCritic(torch.nn.Module):
        def __init__(self, fourier):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(.01))
        def forward(self, y, c):
            return self.weight * y[..., 0]
    generated = []
    def fake_panel(data, n, seed, policy):
        generated.append((seed, policy))
        return {'c': torch.zeros(n, 1), 'positive': torch.ones(n, 3), 'negative': -torch.ones(n, 3)}
    monkeypatch.setattr(audit, 'Critic', TinyCritic)
    monkeypatch.setattr(audit, 'panel', fake_panel)
    monkeypatch.setattr(audit, 'audit_contrasts', lambda predictions: {'verified_keys': list(predictions)})
    emitted = []
    policies = dict.fromkeys(('baseline', *runner.ARMS))
    result = audit.fit_audits(policies, None, tmp_path / 'audit', emitted.append, max_steps=2)
    assert result['state'] == 'budget_exhausted_inconclusive'
    assert len(generated) == 12
    assert all(s['fit_steps'] == 2 and not s['plateau'] for s in result['test'].values())
    baseline = torch.load(tmp_path / 'audit' / 'baseline_best.pt', weights_only=True)
    for arm in runner.ARMS:
        selected = torch.load(tmp_path / 'audit' / f'{arm}_best.pt', weights_only=True)
        assert torch.equal(baseline['model']['weight'], selected['model']['weight'])
    assert all((tmp_path / 'audit' / f'{name}_training_state.pt').is_file() for name in policies)
