import copy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import pytest
import torch

from experiments.dgpo_toy import fourier_film_factorial as factorial
from experiments.dgpo_toy import closed_loop_lab as lab
from experiments.dgpo_toy import conditional as native
from experiments.dgpo_toy.nonperiodic_cube import Critic, Data


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def plan():
    return factorial.load_json(factorial.DEFAULT_PLAN)


def tiny_config(steps=4):
    return native.Config(dimensions=3, context_dim=1, hidden=8, ddim_steps=4,
                         policy_steps=steps, batch=4, candidates=3, timesteps=2,
                         eval_events=8, eval_every=2)


def test_plan_has_one_new_trajectory_and_three_controls():
    p = plan()
    factorial.validate_plan(p)
    assert set(p['conditioning']) == {'raw_shift'}
    assert set(p['factorial']['sources']) == set(factorial.CELLS)-{'raw_shift'}
    assert p['external_control']['arm'] == 'raw'
    assert p['milestones'] == [25, 100, 300, 1000]
    assert p['minimum_fit_steps'] == 8000 and p['max_fit_steps'] == 32000
    assert len(p['run_name']) < 96 and '_' not in p['run_name']


@pytest.mark.parametrize('change', ['basis', 'layers', 'budget', 'reward_control', 'missing_cell'])
def test_plan_rejects_hidden_interventions(change):
    p = plan()
    if change == 'basis':
        p['conditioning']['raw_shift']['basis'] = 'fourier'
    elif change == 'layers':
        p['conditioning']['raw_shift']['layers'] = 1
    elif change == 'budget':
        p['minimum_fit_steps'] = 2000
    elif change == 'reward_control':
        p['external_control']['arm'] = 'fourier'
    else:
        del p['factorial']['sources']['raw_film']
    with pytest.raises(ValueError):
        factorial.validate_plan(p)


def test_contrast_signs_and_additive_null():
    x = torch.arange(32, dtype=torch.float64)
    gains = dict(raw_shift=x, raw_film=x+2, fourier_shift=x+1, fourier_film=x+4)
    contrasts = factorial.factorial_contrasts(gains)
    expected = dict(film_effect_raw=2, film_effect_fourier=3, fourier_effect_shift=1,
                    fourier_effect_film=2, interaction=1)
    for key, value in expected.items():
        assert torch.equal(contrasts[key], torch.full_like(x, value))
    intervals = factorial.paired.paired_intervals(contrasts, repeats=100)
    assert intervals['interaction'] == {'mean': 1., 'lo95': 1., 'hi95': 1.}
    gains['fourier_film'] = x+3
    assert not factorial.factorial_contrasts(gains)['interaction'].any()


def test_invalid_unpaired_or_nonfinite_contrasts():
    gains = {key: torch.zeros(8) for key in factorial.CELLS}
    gains['raw_shift'] = torch.zeros(7)
    with pytest.raises(ValueError):
        factorial.factorial_contrasts(gains)
    gains['raw_shift'] = torch.full((8,), float('nan'))
    with pytest.raises(ValueError):
        factorial.factorial_contrasts(gains)


def test_decision_distinguishes_small_from_unresolved():
    fn = lambda lo, hi: factorial.classify_interval({'lo95': lo, 'hi95': hi}, .01, interaction=True)
    assert fn(.02, .04) == 'supports_positive_interaction'
    assert fn(-.04, -.02) == 'supports_negative_interaction'
    assert fn(-.005, .005) == 'interaction_within_declared_small_effect_band'
    assert fn(-.05, .05) == 'unresolved'
    assert fn(.001, .02) == 'unresolved'


def test_initial_match_and_raw_shift_scale_disabled_after_learning():
    cfg = tiny_config()
    with torch.random.fork_rng():
        torch.manual_seed(17)
        source = native.Denoiser(cfg)
        models = {}
        for cell in factorial.CELLS:
            torch.manual_seed(451017)
            models[cell] = lab.ConditioningAblation(cfg, source.state_dict(), **factorial.architecture(cell))
    matched = lab.film.previous.verify_matching(source, models, cfg)
    assert len({r['parameters'] for r in matched.values()}) == 1
    model = models['raw_shift']
    with torch.no_grad():
        for head in model.condition_heads:
            head.weight.fill_(.1)
            head.bias.fill_(.2)
    pairs, _ = model.modulation(torch.linspace(-1, 1, 8)[:, None], torch.ones(8)*.3)
    assert all(torch.count_nonzero(gamma) == 0 for gamma, _ in pairs)
    assert any(torch.count_nonzero(beta) > 0 for _, beta in pairs)


def test_raw_shift_audit_pause_does_not_change_native_update_trajectory():
    cfg = tiny_config()
    with torch.random.fork_rng():
        torch.manual_seed(17)
        source = native.Denoiser(cfg)
        model = lab.ConditioningAblation(cfg, source.state_dict(), **factorial.architecture('raw_shift'))
        reward = Critic(True).eval().requires_grad_(False)
    saved = {}
    def checkpoint(step, m, opt, rng, history):
        saved.update(model=copy.deepcopy(m.state_dict()), optimizer=copy.deepcopy(opt.state_dict()),
                     rng=rng.get_state(), step=step, history=copy.deepcopy(history), velocity_coefficient=1.)
    before = torch.get_rng_state().clone()
    full, history = native.policy_train('dgpo', model, reward, Data(cfg), cfg, 17, 37,
                                       lambda _: None, velocity_coefficient=1.)
    native.policy_train('dgpo', model, reward, Data(cfg), factorial.replace(cfg, policy_steps=2),
                        17, 37, lambda _: None, checkpoint, velocity_coefficient=1.)
    resumed, resumed_history = native.policy_train('dgpo', model, reward, Data(cfg), cfg, 17, 37,
        lambda _: None, resume_state=saved, velocity_coefficient=1.)
    assert history == resumed_history
    assert torch.equal(before, torch.get_rng_state())
    assert all(torch.equal(v, resumed.state_dict()[k]) for k, v in full.state_dict().items())


def test_record_rejects_config_reward_and_architecture_changes():
    p, cfg = plan(), tiny_config()
    cell, arm = 'raw_film', 'raw'
    old = {**p, 'conditioning': {arm: factorial.architecture(cell)}}
    report = {'state': 'completed', 'valid': True, 'config': asdict(cfg), 'velocity_coefficient': 1.,
              'reward': str(lab.HISTORY/'rewards/full_reward.pt')}
    with tempfile.TemporaryDirectory(dir=lab.ARTIFACTS, prefix='factorial_validation_') as tmp:
        folder = Path(tmp)
        (folder/'report.json').write_text(json.dumps(report))
        (folder/'plan.json').write_text(json.dumps(old))
        spec = {'directory': str(folder), 'arm': arm}
        for key, value in [('reward', '/wrong/reward.pt'), ('velocity_coefficient', .5),
                           ('config', {**asdict(cfg), 'policy_lr': .9})]:
            changed = {**report, key: value}
            (folder/'report.json').write_text(json.dumps(changed))
            with pytest.raises(ValueError):
                factorial.checked_record(cell, spec, p, cfg)
        (folder/'report.json').write_text(json.dumps(report))
        old['conditioning'][arm]['nonlinear'] = False
        (folder/'plan.json').write_text(json.dumps(old))
        with pytest.raises(ValueError, match='architecture'):
            factorial.checked_record(cell, spec, p, cfg)


def test_preflight_checks_real_controls_without_training_or_writes():
    output = lab.ROOT / plan()['factorial']['output']
    existed = output.exists()
    before = torch.get_rng_state().clone()
    with patch.object(lab, 'run', side_effect=AssertionError('preflight must not train')):
        result = factorial.preflight(plan())
    assert result['training_started'] is False
    assert result['new_policy_trajectories'] == 1 and len(result['reused_controls']) == 3
    assert result['test_contexts'] == 16384
    assert all(p['velocity_exact'] and p['samples_exact'] for p in result['initial_matching'].values())
    assert output.exists() == existed and torch.equal(before, torch.get_rng_state())


def test_wrapper_launches_only_missing_cell_and_stops_on_invalid_audit():
    with tempfile.TemporaryDirectory(dir=lab.ARTIFACTS, prefix='factorial_wrapper_') as tmp:
        output = Path(tmp)/'run'
        with patch.object(factorial, 'preflight', return_value={'training_started': False}), \
             patch.object(lab, 'run', return_value={'state': 'inconclusive_audit_fit', 'valid': False}) as launch, \
             patch.object(factorial, 'analyze', side_effect=AssertionError('do not analyze incomplete result')):
            result = factorial.run(factorial.DEFAULT_PLAN, output, 'disabled')
        assert launch.call_count == 1
        assert launch.call_args.args[1] == output/'raw_shift'
        assert set(json.loads(launch.call_args.args[0].read_text())['conditioning']) == {'raw_shift'}
        assert result['state'] == 'inconclusive_audit_fit'
        assert factorial.load_json(output/'status.json')['state'] == 'awaiting_review'
        with patch.object(lab, 'run', side_effect=AssertionError('analysis-only must not train')), \
             patch.object(factorial, 'analyze', return_value={'state': 'completed', 'decisions': {}}) as analyze:
            factorial.run(factorial.DEFAULT_PLAN, output, 'disabled', analysis_only=True)
        analyze.assert_called_once()


def test_analysis_and_plots_end_to_end_on_synthetic_saved_panels():
    """Exercise all four cell reads/contrasts/report/plot without any model fitting."""
    p = plan()
    c = torch.linspace(-1, 1, 32)[:, None]
    base = {split: {'c': c.clone(), 'positive': torch.zeros(32, 3),
                    'negative': torch.zeros(32, 3)} for split in ('train', 'validation', 'test')}
    base_grid = {'contexts': c[:2], 'rewards': {'full': torch.zeros(2, 4)}}
    audit = {'auc': .63, 'bce': .66, 'valid': True, 'plateau': True, 'fit_steps': 8000}
    final_gains = dict(raw_shift=.1, raw_film=.3, fourier_shift=.2, fourier_film=.6)
    class LinearTestReward(torch.nn.Module):
        def __init__(self, unused):
            super().__init__()
        def load_state_dict(self, unused):
            pass
        def forward(self, y, context):
            return y[:, 0]
    with tempfile.TemporaryDirectory(dir=lab.ARTIFACTS, prefix='factorial_analysis_test_') as tmp:
        output, history = Path(tmp)/'analysis', Path(tmp)/'history'
        output.mkdir()
        (history/'rewards').mkdir(parents=True)
        torch.save({'model': {}}, history/'rewards/full_reward.pt')
        records = {}
        for cell in factorial.CELLS:
            folder = Path(tmp)/cell
            folder.mkdir()
            old = {'baseline': {'mode_tv': .2}, 'audits': {'0': {'test': {'baseline__both': audit}}},
                   'points': {}}
            for step in p['milestones']:
                gain = final_gains[cell]*step/1000
                pp = copy.deepcopy(base)
                for split in pp.values():
                    split['negative'][:, 0] = gain
                torch.save(pp, folder/f'{cell}_step{step}_panels.pt')
                grid = {'contexts': c[:2], 'rewards': {'full': torch.full((2, 4), gain)}}
                torch.save(grid, folder/f'{cell}_endpoint{step}.pt')
                old['points'][f'{cell}_{step}'] = {
                    'metrics': {'mode_tv': .2-gain*.01},
                    'reward_gain': {'gain': float(grid['rewards']['full'].double().mean())},
                    'reward_decomposition': {}, 'conditioning': {'fixed_probe_velocity_mse': .001*step/1000}}
                old['audits'][str(step)] = {'test': {cell+'__both': audit}}
            records[cell] = {'folder': folder, 'arm': cell, 'report': old}
        with patch.object(factorial, 'matched_sources', return_value=(tiny_config(), None, records, base, base_grid)), \
             patch.object(factorial, 'Critic', LinearTestReward), patch.object(lab, 'HISTORY', history):
            result = factorial.analyze(p, output)
        assert result['state'] == 'completed' and len(result['primary']) == 5
        assert result['primary']['interaction']['mean'] == pytest.approx(.2)
        assert result['decisions']['interaction'] == 'supports_positive_interaction'
        assert 'Test reward:32 paired IID contexts' in (output/'SUMMARY.md').read_text()
        for name in ('report.json', 'paired_reward_gains.pt', 'factorial_curves.png'):
            assert (output/name).stat().st_size > 0
        saved = torch.load(output/'paired_reward_gains.pt', map_location='cpu', weights_only=True)
        assert torch.equal(saved['contexts'], c)
        assert set(saved['gains']) == set(factorial.CELLS)
