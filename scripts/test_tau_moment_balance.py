"""Local small-array tests; never submit real inference or remote training."""
import json
from unittest.mock import patch

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative

from scripts.tau_moment_balance import (
    CalibrationObjective, analyze, candidate_features, event_split, fit,
    guardrails, metrics, select, transform,
)
from scripts.diagnose_tau_moment_balance import ROOT, read_settings, plots, execute, publish
from scripts.tau_tail_attribution import candidate_weights


def fixture(n=180, k=8):
    rng = np.random.default_rng(14)
    g = rng.normal(size=(n, k, 9))
    inputs = dict(source_ids=np.arange(n).astype(str), weight=rng.uniform(.5, 1.5, n),
        truth_cij=g.mean(1)+np.array([0, .2, 0, 0, 0, 0, 0, 0, 0]),
        category=np.resize([11, 12, 21, 22], n), visible_pt_sum=np.arange(n, dtype=float))
    s = np.minimum(rng.normal(0, .3, (n, k)), np.log(30)-.01)
    cfg = read_settings(ROOT/'config/conditional_tau_moment_balance.yaml')
    cfg.update(condition_pt_edges=[40, 100], minimum_split_events=1, bootstrap=50,
        max_iterations=80, moment_strengths=[.1, 1., 10.])
    return inputs, g, s, cfg


def test_split_unique_order_invariant_and_candidate_clustered():
    ids = np.arange(1000).astype(str)
    a = event_split(ids, 123, [.4, .2, .4])
    order = np.random.default_rng(1).permutation(len(ids))
    np.testing.assert_array_equal(a[order], event_split(ids[order], 123, [.4, .2, .4]))
    assert set(a) == {0, 1, 2}
    assert np.all(np.bincount(a) > 100)
    with pytest.raises(ValueError):
        event_split(['a', 'a'], 123, [.4, .2, .4])


def test_identity_and_bounds_and_nonuniform_base():
    inputs, g, s, cfg = fixture()
    center, scale = candidate_features(g, inputs['weight'])
    model = dict(center=center, scale=scale, theta=np.zeros(9), cap=30, max_log_correction=np.log(2))
    np.testing.assert_array_equal(transform(s, g, model), s)
    model['theta'] = np.arange(9)*10
    out = transform(s, g, model)
    assert np.max(out) <= np.log(30)
    assert np.max(np.abs(out-s)) <= np.log(2)+1e-12
    w, _ = candidate_weights(inputs['weight'], out)
    direct = inputs['weight'][:, None]*np.exp(out)
    direct /= direct.sum()
    np.testing.assert_allclose(w, direct, atol=1e-15)
    assert np.isclose(w.sum(), 1)


@pytest.mark.parametrize('mass_strength', [0., 1.])
def test_analytic_gradient_through_normalization_KL_cap_and_mass(mass_strength):
    inputs, g, s, cfg = fixture(36, 5)
    cfg['mass_strength'] = mass_strength
    s[:8] = np.log(30)-.01
    obj = CalibrationObjective(inputs, g, s, cfg, 1.3)
    theta = np.linspace(-.15, .18, 9)
    value, grad = obj(theta)
    numerical = approx_derivative(lambda x: obj(x)[0], theta, method='3-point').ravel()
    np.testing.assert_allclose(grad, numerical, rtol=2e-5, atol=3e-7)
    assert np.isfinite(value)


def test_zero_strength_identity_and_known_shift_fitting():
    inputs, g, s, cfg = fixture(400, 16)
    s[:] = 0
    identity = fit(inputs, g, s, cfg, 0.)
    assert identity['converged']
    np.testing.assert_allclose(identity['theta'], 0, atol=1e-8)
    model = fit(inputs, g, s, cfg, 10.)
    assert model['converged']
    before = metrics(inputs, g, s, cfg['condition_pt_edges'])[0]
    after = metrics(inputs, g, transform(s, g, model), cfg['condition_pt_edges'])[0]
    assert after['offdiagonal_error'] < .3*before['offdiagonal_error']
    assert after['candidate_ess_fraction'] > .8
    # Parameters round-trip to a deployable transform; no truth needed.
    reloaded = json.loads(json.dumps(model, allow_nan=False))
    np.testing.assert_array_equal(transform(s, g, model), transform(s, g, reloaded))


def test_test_truth_does_not_enter_fit_selection_or_transform():
    inputs, g, s, cfg = fixture()
    events = []
    def seal(frozen, split):
        assert 'test' not in frozen
        assert not frozen['selection']['used_test']
        events.append('seal')
    def progress(stage, values):
        if stage == 'heldout_test':
            assert events == ['seal']
    a, arrays = analyze(inputs, g, s, cfg, progress, seal)
    changed = dict(inputs, truth_cij=inputs['truth_cij'].copy())
    changed['truth_cij'][arrays['event_split'] == 2] += 100
    b, _ = analyze(changed, g, s, cfg)
    assert a['models'] == b['models']
    assert a['selection'] == b['selection']
    assert a['validation'] == b['validation']
    assert a['test']['arms']['baseline']['error'] != b['test']['arms']['baseline']['error']
    assert arrays['bootstrap_C'].shape == (50, 3, 9)
    np.testing.assert_allclose(arrays['event_masses'].sum(0), 1)
    json.dumps(a, allow_nan=False)


def test_selection_guardrails_nonconvergence_and_fallback():
    _, _, _, cfg = fixture()
    base = dict(offdiagonal_error=.2, diagonal_error=.1, error=.23,
        candidate_ess_fraction=.08, event_ess_fraction=.6, group_mass_tv=.1)
    good = dict(base, offdiagonal_error=.1, error=.15)
    bad = dict(good, diagonal_error=.3)
    models = {'low': dict(converged=True, strength=.1), 'high': dict(converged=False, strength=10.)}
    assert select(models, dict(low=good, high=good), base, cfg)['selected'] == 'low'
    assert select(models, dict(low=bad, high=good), base, cfg)['selected'] == 'baseline'
    assert not guardrails(dict(good, candidate_ess_fraction=.01), base, cfg)['candidate_ess']
    assert not guardrails(dict(good, group_mass_tv=.12), base, cfg)['condition_mass']


def test_constant_columns_and_extreme_finite_scores():
    inputs, g, s, cfg = fixture()
    g[..., 3:] = 0
    inputs['truth_cij'][..., 3:] = 0
    s[:, 0] = -1000
    obj = CalibrationObjective(inputs, g, s, cfg, 1.)
    value, grad = obj(np.zeros(9))
    assert np.isfinite(value) and np.isfinite(grad).all()
    np.testing.assert_allclose(grad[3:], 0, atol=1e-10)


def test_source_contract_and_pipeline_artifacts(tmp_path):
    cfg0 = read_settings(ROOT/'config/conditional_tau_moment_balance.yaml')
    assert cfg0['source_workers'] == 16 and cfg0['A']['run'] == 'zrv2yfgt'
    assert 'B' not in cfg0
    assert cfg0['split_fractions'] == [.4, .2, .4]
    inputs, g, s, cfg = fixture()
    w, lm = candidate_weights(inputs['weight'], s)
    truth = np.average(inputs['truth_cij'], weights=inputs['weight'], axis=0).reshape(3, 3)
    c = np.einsum('nk,nkd->d', w, g).reshape(3, 3)
    saved = dict(truth_C=truth, arms={'bounded': dict(C=c, error=np.linalg.norm(c-truth),
        event_ess=1/np.square(w.sum(1)).sum(), candidate_ess=1/np.square(w).sum(),
        max_candidate_mass=w.max(), log_mean_ratio=lm)})
    np.savez(tmp_path/'inputs.npz', visible_pt_sum=inputs['visible_pt_sum'])
    cfg.update(source=str(tmp_path), train_events='filtered', A={})
    with patch('scripts.diagnose_tau_moment_balance.load_source', return_value=({}, inputs, {'cij': g}, {}, {})), \
         patch('scripts.diagnose_tau_moment_balance.read_model', return_value=({'condition_pt_edges': [40, 100]}, s, saved)), \
         patch('evenet_dgpo.evenet.dataset.filtered_data.validate_filtered_dataset', return_value={'rows': 416701}):
        report = execute(cfg, tmp_path)
    for file in ('COMPLETE', 'moment_balance_report.json', 'calibration_model.json', 'split_ids.npz',
                 'heldout_event_bootstrap.npz', 'heldout_moments.png', 'calibration_selection.png'):
        assert (tmp_path/file).stat().st_size > 0
    assert report['source_endpoints_verified'] and not report['pristine_test']
    assert report['classifier_fits'] == 0 and report['policy_updates'] == 0
    assert report['calibration_fits'] == 3
    frozen = json.loads((tmp_path/'calibration_model.json').read_text())
    assert 'test' not in frozen and frozen['selection'] == report['selection']
    with np.load(tmp_path/'heldout_event_bootstrap.npz') as f:
        assert len(f['source_ids']) == report['split_counts']['test']
    class Run:
        def __init__(self): self.summary, self.logs, self.files = {}, [], []
        def log(self, values): self.logs.append(values)
        def save(self, path, **kwargs): self.files.append(path)
    run = Run()
    with patch('wandb.Table', side_effect=lambda **kw: kw), patch('wandb.Image', side_effect=lambda path: path):
        publish(run, report, tmp_path)
    assert run.summary['phase'] == 'complete'
    assert 'test/offdiagonal/lo95' in run.summary
    assert len(run.files) == 5


def test_invalid_inputs_and_empty_splits_fail():
    inputs, g, s, cfg = fixture()
    with pytest.raises(ValueError): analyze(dict(inputs, weight=np.zeros(len(s))), g, s, cfg)
    with pytest.raises(ValueError): analyze(dict(inputs, source_ids=np.zeros(len(s))), g, s, cfg)
    with pytest.raises(ValueError): analyze(inputs, g, s+100, cfg)
    with pytest.raises(ValueError): analyze(inputs, g, s, dict(cfg, minimum_split_events=10000))
