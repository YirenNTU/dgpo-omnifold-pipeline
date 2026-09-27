import copy
import json
from dataclasses import asdict, replace

import pytest
import torch

from experiments.dgpo_toy import conditional as native
from experiments.dgpo_toy import conditioning_transport as runner
from experiments.dgpo_toy import reward_transport as transport
from experiments.dgpo_toy.cube_lockdown import ConditionDenoiser
from experiments.dgpo_toy.nonperiodic_cube import Critic, Data


def config():
    return native.Config(dimensions=3, context_dim=1, hidden=8, ddim_steps=50,
                         policy_steps=2, eval_every=100, eval_events=512,
                         batch=4, candidates=3, timesteps=2)


def test_routes_have_exact_initial_functions_and_honest_parameter_counts():
    cfg = config()
    initial = native.Denoiser(cfg).eval()
    rng = torch.get_rng_state().clone()
    models = runner.make_models(initial, cfg)
    assert torch.equal(rng, torch.get_rng_state())
    matches = runner.verify_matching(initial, models, cfg)
    assert [matches[n]['added_parameters'] for n in runner.ARMS] == [64, 192, 384]
    assert all(v['velocity_exact'] and v['samples_exact'] for v in matches.values())
    assert isinstance(models['first_add'], ConditionDenoiser)
    assert runner.equal_state(models['first_add'].network.state_dict(), initial.network.state_dict())
    for m in models.values():
        assert all(torch.count_nonzero(v) == 0 for k, v in m.state_dict().items() if not k.startswith('network.'))


@pytest.mark.parametrize('name', runner.ARMS)
def test_every_route_works_with_native_loss_and_has_live_condition_gradients(name):
    cfg = replace(config(), ddim_steps=2)
    initial = native.Denoiser(cfg)
    model = runner.make_models(initial, cfg)[name]
    reference = copy.deepcopy(model).requires_grad_(False)
    reward = transport.ModeTableReward(torch.tensor([-1., 1.]), torch.full((2, 8), .125), oracle=True)
    data, rng = Data(cfg), native.generator(91)
    c = data.contexts(cfg.batch, rng)
    z = torch.randn(cfg.batch, cfg.candidates, 3, generator=rng)
    t = .7*torch.rand(cfg.timesteps, cfg.batch, generator=rng)
    eps = torch.randn(cfg.timesteps, cfg.batch, 3, generator=rng)
    loss, stats = native.dgpo_objective(model, reference, reward, data, cfg, c, z, t, eps,
                                      velocity_coefficient=1., trace_gradients=True)
    loss.backward()
    assert stats['velocity_mse'] == 0 and stats['velocity_coefficient'] == 1
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    for key, p in model.named_parameters():
        if not key.startswith('network.'):
            assert p.grad.norm() > 0, key
    probe = runner.condition_probe(initial, cfg)
    diagnostics = runner.route_diagnostics(model, initial, probe)
    assert diagnostics['fixed_probe_velocity_mse'] == 0
    assert all(v['residual_over_hidden_rms'] == 0 for v in diagnostics['layers'].values())
    assert diagnostics['last_step_postclip_gradient_norm']['condition'] > 0
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.policy_lr, weight_decay=cfg.weight_decay)
    opt.step()  # One tiny synthetic unit-test update, not an experimental arm.
    diagnostics = runner.route_diagnostics(model, initial, probe)
    assert diagnostics['fixed_probe_velocity_mse'] > 0
    assert all(v['residual_over_hidden_rms'] > 0 for v in diagnostics['layers'].values())
    if name == 'deep_film':
        assert all(v['scale_rms'] > 0 for v in diagnostics['layers'].values())


def test_deep_add_and_film_can_reproduce_trained_first_layer_route():
    cfg = config()
    initial = native.Denoiser(cfg)
    models = runner.make_models(initial, cfg)
    with torch.no_grad():
        models['first_add'].condition_adapter.weight.normal_(std=.05)
        for name in runner.ARMS[1:]:
            models[name].condition_adapter.weight.copy_(models['first_add'].condition_adapter.weight)
    rng = native.generator(4)
    x = torch.randn(2, 3, 4, 3, generator=rng)
    c = torch.rand(2, 1, 4, 1, generator=rng)*2-1
    t = torch.rand(2, 1, 4, generator=rng)
    v, h = models['first_add'].predict_features(x, t, c)
    for name in runner.ARMS[1:]:
        got, hidden = models[name].predict_features(x, t, c)
        assert torch.equal(got, v) and torch.equal(hidden, h)
    with pytest.raises(ValueError):
        runner.DeepConditionDenoiser(cfg, 'invalid', initial.state_dict())


def test_probe_diagnostics_do_not_consume_global_rng_or_modify_weights():
    cfg = config()
    initial = native.Denoiser(cfg)
    model = runner.make_models(initial, cfg)['deep_film']
    state = copy.deepcopy(model.state_dict())
    rng = torch.get_rng_state().clone()
    probe = runner.condition_probe(initial, cfg)
    runner.route_diagnostics(model, initial, probe)
    assert torch.equal(rng, torch.get_rng_state())
    assert runner.equal_state(model.state_dict(), state)


def write_previous(tmp_path):
    cfg = config()
    source, cls, previous = (tmp_path/name for name in ('reference', 'classifier', 'previous'))
    for p in (source, cls, previous):
        p.mkdir()
    initial = native.Denoiser(cfg)
    torch.save({'model': initial.state_dict(), 'config': asdict(cfg),
                'trained_on': 'uniform cube reference, not full truth'}, source/'reference.pt')
    ck = {'model': Critic(True).state_dict(), 'selected_step': 3100}
    torch.save(ck, cls/'classifier.pt')
    torch.save(ck, previous/'classifier.pt')
    (cls/'report.json').write_text(json.dumps({
        'state': 'completed', 'source': str(source), 'config': asdict(cfg),
        'velocity_coefficient': 1., 'classifier': {'gate': {'passed': True}, 'selected_step': 3100}}))
    (previous/'report.json').write_text(json.dumps({
        'state': 'completed', 'source': str(source), 'classifier_source': str(cls),
        'config': asdict(cfg), 'velocity_coefficient': 1., 'basis': 'fourier',
        'seeds': transport.SEEDS, 'calibration': {'passed': True}}))
    torch.save({'nodes': torch.tensor([-1., 1.]), 'mean_scores': torch.zeros(2, 8),
                'q_initial': torch.full((2, 8), .125)}, previous/'calibration.pt')
    torch.save({'model': runner.make_models(initial, cfg)['first_add'].state_dict()}, previous/'C_last.pt')
    return previous


def test_cached_reward_lineage_and_no_new_fits(tmp_path, monkeypatch):
    directory = write_previous(tmp_path)
    monkeypatch.setattr(transport, 'calibrate', lambda *a, **kw: pytest.fail('Must reuse calibration'))
    cfg, initial, rewards, prior = runner.load_previous(directory, 1000)
    assert cfg.policy_steps == 1000 and rewards['C'].oracle
    assert all(not p.requires_grad for p in rewards['A'].parameters())
    ck = torch.load(directory/'classifier.pt', weights_only=True)
    ck['model'][next(iter(ck['model']))].add_(.1)
    torch.save(ck, directory/'classifier.pt')
    with pytest.raises(ValueError, match='classifier does not match'):
        runner.load_previous(directory, 1000)


def test_no_closure_claim_without_valid_audit_and_no_short_budget_claim():
    report = {'baseline': {'mode_tv': .2, 'corner_fraction': .95},
              'config': {'policy_steps': 1000}, 'arms': {}, 'contrasts': {}}
    for name, tv in zip(runner.ARMS, [.2, .195, .1]):
        report['arms'][name] = {'endpoint': {'mode_tv': tv, 'corner_fraction': .95, 'missing_mode_cells': 0}}
        if name != 'first_add':
            report['contrasts'][f'{name}_minus_first_add'] = {'delta': tv-.2, 'hi95': tv-.19}
    result = runner.decisions(report)
    assert result['route_beats_first_add'] == {'deep_add': False, 'deep_film': True}
    assert result['interpretation'].startswith('supports_conditioning')
    assert not result['fresh_audit_valid'] and not any(result['fresh_auc_gap_improves'].values())
    report['config']['policy_steps'] = 300
    assert runner.decisions(report)['interpretation'] == 'short_budget_inconclusive'
    report['config']['policy_steps'] = 1000
    report['arms']['deep_film']['endpoint']['corner_fraction'] = .1
    assert not runner.decisions(report)['route_beats_first_add']['deep_film']


@pytest.mark.parametrize('skip_audit', [False, True])
def test_runner_reuses_frozen_C_and_same_native_configuration(tmp_path, monkeypatch, skip_audit):
    directory = write_previous(tmp_path)
    endpoint = {'q': torch.full((4, 8), .125), 'mode_tv_by_context': torch.full((4,), .2),
                'rewards': {k: torch.zeros(4, 8) for k in ('A', 'B', 'C')}}
    stats = {'mode_tv': .2, 'corner_fraction': .95, 'missing_mode_cells': 0}
    torch.save(endpoint, directory/'baseline_endpoint.pt')
    monkeypatch.setattr(transport, 'assess', lambda *a, **kw: (copy.deepcopy(stats), copy.deepcopy(endpoint)))
    monkeypatch.setattr(transport, 'calibrate', lambda *a, **kw: pytest.fail('No recalibration'))
    calls = []

    def fake_policy(arm, initial, reward, data, cfg, seed, monitor_seed, emit, callback, **kwargs):
        calls.append((reward, cfg, seed, monitor_seed, kwargs))
        assert arm == 'dgpo' and reward.oracle
        opt = torch.optim.AdamW(initial.parameters(), lr=cfg.policy_lr)
        history = [{'step': i+1} for i in range(cfg.policy_steps)]
        callback(cfg.policy_steps, initial, opt, native.generator(8), history)
        return initial, history

    def fake_audit(policies, data, output, emit, max_steps):
        assert list(policies) == ['baseline', *runner.ARMS] and max_steps == 16000
        output.mkdir()
        torch.save({k: {'positive': torch.zeros(8), 'negative': torch.zeros(8)} for k in policies},
                   output/'test_scores.pt')
        return {'state': 'completed', 'contrasts': {
            f'{k}_minus_baseline': {'auc_gap': {'hi95': .01}} for k in runner.ARMS}}

    monkeypatch.setattr(native, 'policy_train', fake_policy)
    monkeypatch.setattr(runner, 'fit_audits', fake_audit)
    result = runner.run(directory, tmp_path/'output', steps=2, wandb_mode='disabled', skip_audit=skip_audit)
    assert len(calls) == 3 and len({id(c[0]) for c in calls}) == 1
    assert all(c[2:] == (17, 284017, {'velocity_coefficient': 1.}) for c in calls)
    assert result['first_add_replays_previous_C_exactly'] and result['cached_baseline_exact']
    assert result['source_unchanged'] and all(result['rewards_unchanged'].values())
    assert result['state'] == ('completed_without_audit' if skip_audit else 'completed')
    if not skip_audit:
        assert 'deep_film_minus_first_add' in result['audit']['route_contrasts']


def test_incompatible_cached_baseline_stops_before_any_training(tmp_path, monkeypatch):
    directory = write_previous(tmp_path)
    torch.save({'q': torch.zeros(4, 8), 'rewards': {}}, directory/'baseline_endpoint.pt')
    monkeypatch.setattr(transport, 'assess', lambda *a, **kw: ({}, {
        'q': torch.ones(4, 8), 'rewards': {}}))
    monkeypatch.setattr(native, 'policy_train', lambda *a, **kw: pytest.fail('Must stop before updates'))
    with pytest.raises(ValueError, match='not replayable'):
        runner.run(directory, tmp_path/'output', steps=2, wandb_mode='disabled')
    report = json.loads((tmp_path/'output/report.json').read_text())
    assert report['state'] == 'failed_or_interrupted'


def test_protocol_stays_local_and_does_not_leak_truth_into_model():
    import inspect
    from pathlib import Path
    source = inspect.getsource(runner.DeepConditionDenoiser)
    assert 'condition_signal' not in source and 'probabilities' not in source and 'modes(' not in source
    excludes = Path('NERSC/upload-excludes.txt').read_text()
    assert 'experiments/dgpo_toy/' in excludes and 'artifacts/dgpo_toy/' in excludes
