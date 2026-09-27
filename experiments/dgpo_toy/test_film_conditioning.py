import copy
import json
from dataclasses import asdict, replace

import pytest
import torch

from experiments.dgpo_toy import conditional as native
from experiments.dgpo_toy import film_conditioning as runner
from experiments.dgpo_toy import reward_transport as transport
from experiments.dgpo_toy.nonperiodic_cube import Critic, Data
from experiments.dgpo_toy.test_conditioning_transport import config, write_previous


def test_initial_function_and_common_encoder_weights_match():
    cfg = config()
    initial = native.Denoiser(cfg)
    rng = torch.get_rng_state().clone()
    models = runner.make_models(initial, cfg, width=8)
    assert torch.equal(rng, torch.get_rng_state())
    match = runner.previous.verify_matching(initial, models, cfg)
    assert all(m['velocity_exact'] and m['samples_exact'] for m in match.values())
    assert match['mlp_ct']['parameters']-match['mlp_c']['parameters'] == 5*8
    c_state, ct_state = models['mlp_c'].state_dict(), models['mlp_ct'].state_dict()
    assert all(torch.equal(v, ct_state[k]) for k, v in c_state.items())
    assert c_state['condition_input.weight'].norm() > 0
    assert all(p.norm() == 0 for p in models['mlp_c'].condition_heads.parameters())
    assert all(not k.startswith('network.') or torch.equal(v, initial.state_dict()[k]) for k, v in c_state.items())
    learned_models = runner.make_models(initial, cfg, width=8, arms=runner.LEARNED_ARMS)
    assert list(learned_models) == list(runner.LEARNED_ARMS)
    assert all(runner.previous.equal_state(m.state_dict(), models[name].state_dict())
               for name, m in learned_models.items())


def test_time_only_modifies_coefficients_in_ct_arm():
    cfg = config()
    models = runner.make_models(native.Denoiser(cfg), cfg, width=8)
    c = torch.linspace(-.8, .8, 6)[:, None]
    t1, t2 = torch.full((6,), .1), torch.full((6,), .6)
    for name in ('mlp_c', 'mlp_ct'):
        model = models[name]
        with torch.no_grad():
            for h in model.condition_heads:
                h.weight.fill_(0.)
                h.weight[:, 0] = .05
            first, _ = model.modulation(c, t1)
            second, _ = model.modulation(c, t2)
        same = all(torch.equal(a, b) for la, lb in zip(first, second) for a, b in zip(la, lb))
        assert same == (name == 'mlp_c')
        # Backbone always sees t, even when modulation does not.
        assert not torch.equal(model(torch.ones(6, 3), t1, c), model(torch.ones(6, 3), t2, c))


@pytest.mark.parametrize('name', ['mlp_c', 'mlp_ct'])
def test_zero_heads_learn_first_then_encoder_receives_gradient(name):
    cfg = replace(config(), ddim_steps=2)
    with torch.random.fork_rng():
        torch.manual_seed(17)
        initial = native.Denoiser(cfg)
    model = runner.make_models(initial, cfg, width=8)[name]
    reference = copy.deepcopy(model).requires_grad_(False)
    data, rng = Data(cfg), native.generator(91)
    class GradientProbeReward(torch.nn.Module):
        def forward(self, y, c, data=None):
            return y[..., 0] + .5*c[..., 0]*y[..., 1]
    # This is a gradient-connectivity unit test, not a mode-coverage experiment.
    # An untrained 2-step generator may put every candidate in the same parity
    # class: an oracle categorical reward then has exactly zero LOO advantage.
    reward = GradientProbeReward()
    c = data.contexts(cfg.batch, rng)
    z = torch.randn(cfg.batch, cfg.candidates, 3, generator=rng)
    t = .7*torch.rand(cfg.timesteps, cfg.batch, generator=rng)
    eps = torch.randn(cfg.timesteps, cfg.batch, 3, generator=rng)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.policy_lr, weight_decay=cfg.weight_decay)
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        loss, stats = native.dgpo_objective(model, reference, reward, data, cfg, c, z, t, eps,
                                          velocity_coefficient=1., trace_gradients=True)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        assert sum(float(p.grad.square().sum()) for p in model.condition_heads.parameters()) > 0
        if step == 0:
            assert stats['velocity_mse'] == 0 and model.condition_input.weight.grad.norm() == 0
        else:
            assert model.condition_input.weight.grad.norm() > 0
            if name == 'mlp_ct':
                assert model.condition_time.weight.grad.norm() > 0
        opt.step()
    diag = runner.diagnostics(model, initial, runner.previous.condition_probe(initial, cfg))
    assert diag['fixed_probe_velocity_mse'] > 0
    assert diag['last_step_postclip_gradient_norm']['encoder'] > 0
    assert (diag['time_modulation_difference_rms'] > 0) == (name == 'mlp_ct')
    assert set(diag['coefficient_rms_by_time']) == {'t0', 't035', 't070', 't100'}
    if name == 'mlp_c':
        assert len(set(diag['coefficient_rms_by_time'].values())) == 1


@pytest.mark.parametrize('name,reward_arm', [(n, 'C') for n in runner.ARMS] +
                         [(n, 'A') for n in runner.LEARNED_ARMS])
def test_staged_native_resume_exactly_matches_continuous_training(name, reward_arm):
    cfg = replace(config(), ddim_steps=2, policy_steps=4, eval_events=8, eval_every=2)
    with torch.random.fork_rng():
        torch.manual_seed(23)
        initial = native.Denoiser(cfg)
    model = runner.make_models(initial, cfg, width=8)[name]
    data = Data(cfg)
    reward = (transport.ModeTableReward(torch.tensor([-1., 1.]), torch.full((2, 8), .125), oracle=True)
              if reward_arm == 'C' else Critic(True, width=8).eval().requires_grad_(False))
    reward_before = copy.deepcopy(reward.state_dict())
    before = copy.deepcopy(model.state_dict())
    records = []

    def save(step, current, optimizer, rng, history):
        records.append(copy.deepcopy({'model': current.state_dict(), 'optimizer': optimizer.state_dict(),
            'rng': rng.get_state(), 'history': history, 'step': step, 'velocity_coefficient': 1.}))

    full, _ = native.policy_train('dgpo', model, reward, data, cfg, 17, 284017, lambda r: None,
                                  save, velocity_coefficient=1.)
    full_state = records[-1]
    native.policy_train('dgpo', model, reward, data, replace(cfg, policy_steps=2), 17, 284017,
                         lambda r: None, save, velocity_coefficient=1.)
    half = records[-1]
    resumed, history = native.policy_train('dgpo', model, reward, data, cfg, 17, 284017,
        lambda r: None, save, resume_state=half, velocity_coefficient=1.)
    assert len(history) == 4 and [r['step'] for r in history] == [1, 2, 3, 4]
    assert runner.previous.equal_state(full.state_dict(), resumed.state_dict())
    assert torch.equal(full_state['rng'], records[-1]['rng'])
    for key, value in full_state['optimizer']['state'].items():
        for field, tensor in value.items():
            assert torch.equal(tensor, records[-1]['optimizer']['state'][key][field])
    assert runner.previous.equal_state(model.state_dict(), before)
    assert runner.previous.equal_state(reward.state_dict(), reward_before)


def test_milestones_and_invalid_audit_do_not_produce_success():
    for x in ([], [2, 1], [1, 1], [0, 1], [-1], [1.5]):
        with pytest.raises(ValueError):
            runner.validate_milestones(x)
    assert runner.validate_milestones([1000, 2000, 3000]) == (1000, 2000, 3000)
    baseline = {'mode_tv': .2, 'corner_fraction': .95}
    result = {'arms': {name: {'endpoint': {'mode_tv': .1, 'corner_fraction': .95, 'missing_mode_cells': 0}}
                       for name in runner.ARMS}, 'audit': {'state': 'budget_exhausted_inconclusive'}}
    got = runner.decision(baseline, result)
    assert not got['fresh_audit_valid'] and not any(got['combined_pass'].values())
    assert all(got['mode_transport_pass'].values())


@pytest.mark.parametrize('reward_arm', ['A', 'C'])
def test_staged_runner_keeps_reference_reward_rng_and_audit_clocks(tmp_path, monkeypatch, reward_arm):
    from types import SimpleNamespace
    import sys
    spec = runner.experiment_spec(reward_arm)
    arms = spec['arms']
    directory = write_previous(tmp_path)
    cfg, initial, _, prior = runner.previous.load_previous(directory, 2)
    control = tmp_path/'control'; control.mkdir()
    (control/'report.json').write_text(json.dumps({'state': 'completed', 'source': prior['source'],
        'reward_source': str(directory), 'velocity_coefficient': 1., 'config': asdict(cfg)}))
    if reward_arm == 'C':
        torch.save({'model': runner.make_models(initial, cfg, width=8)['linear'].state_dict()}, control/'deep_film_last.pt')
    panel = {'q': torch.full((4, 8), .125), 'mode_tv_by_context': torch.full((4,), .2),
             'rewards': {k: torch.zeros(4, 8) for k in ('A', 'B', 'C')},
             'learned_cells': {'mean': torch.zeros(4, 8), 'mean_available': torch.ones(4, 8, dtype=torch.bool)}}
    stats = {'mode_tv': .2, 'corner_fraction': .95, 'missing_mode_cells': 0}
    torch.save(panel, directory/'baseline_endpoint.pt')
    monkeypatch.setattr(transport, 'assess', lambda *a, **kw: (copy.deepcopy(stats), copy.deepcopy(panel)))
    calls, fits = [], []

    def fake_train(arm, model, reward, data, cfg, seed, monitor, emit, checkpoint, resume_state, **kwargs):
        assert isinstance(reward, Critic if reward_arm == 'A' else transport.ModeTableReward)
        assert not reward.training and all(not p.requires_grad for p in reward.parameters())
        if resume_state:
            assert resume_state['reward_arm'] == reward_arm
        start = resume_state['step'] if resume_state else 0
        calls.append((id(model), id(reward), start, cfg.policy_steps, seed, monitor, kwargs))
        history = copy.deepcopy(resume_state['history']) if resume_state else []
        history.extend({'step': s} for s in range(start+1, cfg.policy_steps+1))
        optimizer = torch.optim.AdamW(model.parameters())
        checkpoint(cfg.policy_steps, model, optimizer, native.generator(2), history)
        return model, history

    def fake_audit(policies, data, output, emit, max_steps):
        assert list(policies) == ['baseline', *arms]
        fits.append(str(output))
        output.mkdir(parents=True)
        torch.save({name: {'positive': torch.zeros(8), 'negative': torch.zeros(8)} for name in policies},
                   output/'test_scores.pt')
        emit({'phase': 'audit', 'arm': 'linear', 'step': 7500})
        return {'state': 'completed', 'contrasts': {f'{name}_minus_baseline': {'auc_gap': {'hi95': .01}}
                                                  for name in arms},
                'test': {name: {'auc': .6, 'bce': .68, 'auc_gap': .1, 'fit_steps': 7500,
                                'selected_step': 5500, 'plateau': True} for name in policies}}

    metadata, metrics, logged, finished = {}, [], [], []
    fake_wb = SimpleNamespace(id='fixed-test-id', dir='fake-files', summary={},
        define_metric=lambda *a, **kw: metrics.append((a, kw)),
        log=lambda row: logged.append(row), finish=lambda **kw: finished.append(kw))

    def fake_init(**kw):
        metadata.update(kw)
        return fake_wb

    monkeypatch.setattr(native, 'policy_train', fake_train)
    monkeypatch.setattr(runner, 'fit_audits', fake_audit)
    monkeypatch.setattr(transport, 'calibrate', lambda *a, **kw: pytest.fail('No recalibration'))
    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(init=fake_init))
    output = tmp_path/'out'
    result = runner.run(directory, control, output, milestones=(2, 4, 6), width=8,
        wandb_mode='offline' if reward_arm == 'A' else 'disabled', run_id='fixed-test-id', reward_arm=reward_arm)
    assert len(calls) == 3*len(arms) and len({c[1] for c in calls}) == 1 and len(fits) == 3
    assert [(c[2], c[3]) for c in calls] == [(0, 2)]*len(arms)+[(2, 4)]*len(arms)+[(4, 6)]*len(arms)
    assert len({c[0] for c in calls}) == len(arms)
    assert all(c[4:] == (17, 284017, {'velocity_coefficient': 1.}) for c in calls)
    assert result['state'] == 'completed' and result['primary_endpoint_step'] == 6
    assert result['source_unchanged']
    assert result['reward_arm'] == reward_arm and result['arms'] == list(arms)
    assert result['classifier_selected_step'] == 3100
    if reward_arm == 'C':
        assert result['linear_replays_previous_film_exactly']
    else:
        assert 'linear_replays_previous_film_exactly' not in result
        frozen = torch.load(output/'frozen_reward.pt', weights_only=True)
        original = torch.load(directory/'classifier.pt', weights_only=True)
        assert runner.previous.equal_state(frozen['model'], original['model'])
        assert 'time_branch_improves_vs_c_only' not in result['decision']
        assert 'primary_pass' in result['decision']
        assert metadata['name'] == runner.LEARNED_RUN_NAME and metadata['id'] == 'fixed-test-id'
        assert metadata['config']['reward_arm'] == 'A'
        assert 'fixed-mode-reward' not in metadata['tags']
        assert metadata['group'] == 'Conditional reward transport'
        assert not any('mlp_ct' in str(m) for m in metrics)
        assert (('policy/mlp_c/*',), {'step_metric': 'policy/mlp_c/step'}) in metrics
        assert (('audit/round6/mlp_c/*',), {'step_metric': 'audit/round6/mlp_c/step'}) in metrics
        assert (('validation/mlp_c/*',), {'step_metric': 'validation/mlp_c/step'}) in metrics
        assert any('audit/round6/linear/step' in row for row in logged)
        assert fake_wb.summary['state'] == 'completed' and finished == [{'exit_code': 0}]
        assert fake_wb.summary['final_audit/mlp_c/auc'] == .6
        assert result['rounds']['6']['arms']['mlp_c']['learned_reward_decomposition']['available']
    assert all(result['rewards_unchanged'].values()) and all(result['initial_models_unchanged'].values())
    rows=[json.loads(s) for s in (output/'progress.jsonl').read_text().splitlines()]
    assert [(r['milestone'], r['step']) for r in rows if r['phase']=='audit'] == [(2, 7500), (4, 7500), (6, 7500)]
    validation = [r for r in rows if r['phase']=='validation' and r['arm']=='mlp_c']
    assert [(r['step'], r['fit_steps']) for r in validation] == [(2, 7500), (4, 7500), (6, 7500)]
    assert all(r['valid'] for r in validation)
    for name in arms:
        state = torch.load(output/f'{name}_last.pt', weights_only=True)
        assert state['step'] == 6 and len(state['history']) == 6
        assert state['reward_arm'] == reward_arm


def test_policy_inputs_do_not_contain_truth_functions():
    import inspect
    code = inspect.getsource(runner.NonlinearFiLM)
    assert 'probabilities' not in code and 'condition_signal' not in code and 'modes(' not in code


def test_learned_reward_primary_is_fresh_audit_not_reward_or_topology():
    baseline = {'mode_tv': .2, 'corner_fraction': .95}
    result = {'arms': {name: {'endpoint': {'mode_tv': .25, 'corner_fraction': .95, 'missing_mode_cells': 0}}
                       for name in runner.LEARNED_ARMS},
              'audit': {'state': 'completed',
                        'contrasts': {f'{name}_minus_baseline': {'auc_gap': {'hi95': -.01}}
                                      for name in runner.LEARNED_ARMS},
                        'route_contrasts': {'mlp_c_minus_linear': {'auc_gap': {'hi95': -.005}}}}}
    got = runner.decision(baseline, result, reward_arm='A')
    assert got['primary_pass']['mlp_c'] and not got['combined_pass']['mlp_c']
    assert got['interpretation'] == 'supports_nonlinear_learned_H4_transfer_at_this_budget'
    result['audit']['state'] = 'budget_exhausted_inconclusive'
    got = runner.decision(baseline, result, reward_arm='A')
    assert not got['primary_pass']['mlp_c'] and got['interpretation'] == 'audit_inconclusive'
    result['audit']['state'] = 'completed'
    result['audit']['route_contrasts']['mlp_c_minus_linear']['auc_gap']['hi95'] = .001
    got = runner.decision(baseline, result, reward_arm='A')
    assert not got['primary_pass']['mlp_c']
    assert got['interpretation'] == 'learned_H4_transfers_but_nonlinear_advantage_unresolved'
    result['audit']['contrasts']['mlp_c_minus_baseline']['auc_gap']['hi95'] = .001
    assert runner.decision(baseline, result, reward_arm='A')['interpretation'] == 'nonlinear_package_not_sufficient_at_this_budget'


def test_learned_objective_uses_no_oracle_and_does_not_refit_critic():
    class NoTruthAccess:
        def __getattr__(self, name):
            pytest.fail(f'Learned reward loss must not use truth lookup: {name}')

    cfg = replace(config(), ddim_steps=2)
    initial = native.Denoiser(cfg)
    model = runner.make_models(initial, cfg, width=8, arms=runner.LEARNED_ARMS)['mlp_c']
    reference = copy.deepcopy(model).requires_grad_(False)
    critic = Critic(True, width=8).eval().requires_grad_(False)
    before = copy.deepcopy(critic.state_dict())
    rng = native.generator(52)
    c = 2*torch.rand(cfg.batch, 1, generator=rng)-1
    noise = torch.randn(cfg.batch, cfg.candidates, 3, generator=rng)
    t = .7*torch.rand(cfg.timesteps, cfg.batch, generator=rng)
    eps = torch.randn(cfg.timesteps, cfg.batch, 3, generator=rng)
    loss, stats = native.dgpo_objective(model, reference, critic, NoTruthAccess(), cfg, c, noise, t, eps,
                                       velocity_coefficient=1., trace_gradients=True)
    loss.backward()
    assert torch.isfinite(loss) and stats['velocity_coefficient'] == 1.
    assert all(p.grad is None for p in critic.parameters())
    assert runner.previous.equal_state(critic.state_dict(), before)
    assert model.condition_heads[0].weight.grad.norm() > 0


def test_learned_preflight_does_not_create_output_or_start_training(tmp_path, monkeypatch, capsys):
    import sys
    directory = write_previous(tmp_path)
    cfg, initial, rewards, prior = runner.previous.load_previous(directory, 3000)
    monkeypatch.setattr(runner, 'load_inputs', lambda *a: (cfg, initial, rewards, prior, {}))
    monkeypatch.setattr(runner, 'run', lambda *a, **kw: pytest.fail('Preflight must not train'))
    output = tmp_path/'not_created'
    monkeypatch.setattr(sys, 'argv', ['film_conditioning', '--reward-source', str(directory),
        '--reward-arm', 'A', '--output', str(output), '--width', '8', '--preflight'])
    runner.main()
    printed = json.loads(capsys.readouterr().out)
    assert printed['state'] == 'preflight_passed_not_started' and not output.exists()
    assert printed['classifier_frozen'] and printed['classifier_selected_step'] == 3100
    assert printed['reward_arm'] == 'A' and printed['wandb_name'] == runner.LEARNED_RUN_NAME
    assert list(printed['initial_matching']) == list(runner.LEARNED_ARMS)
    for reward in ('B', '', 'learned', None):
        with pytest.raises(ValueError, match='Choose A'):
            runner.experiment_spec(reward)
