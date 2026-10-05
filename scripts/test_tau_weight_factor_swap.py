import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.tau_weight_factor_swap import factor_weights, analyze, verify_endpoint, ARMS
from scripts.diagnose_tau_weight_factor_swap import read_settings, verify_protocol, PROTOCOL_KEYS


def fixture():
    rng = np.random.default_rng(19)
    n, k = 60, 8
    inputs = dict(weight=rng.uniform(.5, 1.5, n), source_ids=np.arange(n),
        truth_cij=rng.normal(size=(n, 9)), category=np.arange(n) % 3,
        visible_pt_sum=rng.uniform(0, 100, n))
    generated = rng.normal(size=(n, k, 9))
    a = rng.normal(size=(n, k)); b = rng.normal(size=(n, k))
    cfg = dict(bootstrap=100, bootstrap_seed=17, condition_pt_edges=[25, 50, 75])
    return inputs, generated, a, b, cfg


def test_identical_models():
    inputs, g, a, _, cfg = fixture()
    report, arrays = analyze(inputs, g, a, a, cfg)
    for name in ARMS[:4]:
        np.testing.assert_allclose(report['arms'][name]['C'], report['arms']['AA']['C'])
    for row in report['contrasts'].values():
        assert abs(row['error_change']) < 1e-14
        np.testing.assert_allclose(row['simultaneous_ci95'], 0, atol=1e-14)
    assert arrays['event_numerators'].shape == (60, 5, 9)


def test_only_mass_changes():
    inputs, _, a, _, _ = fixture()
    b = a+np.linspace(-2, 2, len(a))[:, None]
    w, _ = factor_weights(inputs['weight'], a, b)
    np.testing.assert_allclose(w['AA'], w['AB'], atol=1e-15)
    np.testing.assert_allclose(w['BB'], w['BA'], atol=1e-15)
    assert not np.allclose(w['AA'], w['BB'])


def test_only_within_changes():
    inputs, _, a, _, _ = fixture()
    b = a[:, ::-1]
    w, _ = factor_weights(inputs['weight'], a, b)
    np.testing.assert_allclose(w['AA'], w['BA'], atol=1e-15)
    np.testing.assert_allclose(w['BB'], w['AB'], atol=1e-15)
    assert not np.allclose(w['AA'], w['BB'])


def test_preserves_marginals_and_zero_weight_extreme_logits():
    a = np.array([[1000, -1000], [-1000, 1000], [-900, -800.]])
    b = -a
    w, logs = factor_weights(np.array([0., 1., 2.]), a, b)
    for row in w.values():
        assert np.isfinite(row).all()
        assert row.sum() == pytest.approx(1)
        assert np.all(row[0] == 0)
    np.testing.assert_allclose(w['AB'].sum(1), w['AA'].sum(1))
    np.testing.assert_allclose(w['BA'].sum(1), w['BB'].sum(1))
    assert logs['AB'] is None  # Hybrid is not assigned a spurious ratio normalizer.


def test_factorial_accounting_bootstrap_and_endpoint_verification():
    inputs, g, a, b, cfg = fixture()
    report, arrays = analyze(inputs, g, a, b, cfg)
    c = report['contrasts']
    assert c['within_at_mass_A']['error_change']+c['mass_at_within_B']['error_change'] == pytest.approx(c['total_B_minus_A']['error_change'])
    assert c['mass_at_within_A']['error_change']+c['within_at_mass_B']['error_change'] == pytest.approx(c['total_B_minus_A']['error_change'])
    assert c['within_at_mass_B']['error_change']-c['within_at_mass_A']['error_change'] == pytest.approx(c['interaction']['error_change'])
    np.testing.assert_allclose(arrays['event_masses'][:, 0], arrays['event_masses'][:, 1])
    assert report['arms']['AA']['group_mass_tv'] == pytest.approx(report['arms']['AB']['group_mass_tv'])
    assert len(c['within_at_mass_A']['components']) == 9
    saved = dict(truth_C=report['truth_C'], arms=dict(bounded=copy.deepcopy(report['arms']['AA'])))
    verify_endpoint(report['arms']['AA'], report['truth_C'], saved)
    saved['arms']['bounded']['C'][0][0] += .01
    with pytest.raises(ValueError, match='endpoint differs: C'):
        verify_endpoint(report['arms']['AA'], report['truth_C'], saved)
    json.dumps(report, allow_nan=False)
    repeat, repeat_arrays = analyze(inputs, g, a, b, cfg)
    np.testing.assert_array_equal(arrays['bootstrap_C'], repeat_arrays['bootstrap_C'])
    assert report == repeat


@pytest.mark.parametrize('kind', ['nan', 'duplicate', 'shape'])
def test_reject_invalid_panel(kind):
    inputs, g, a, b, cfg = fixture()
    if kind == 'nan': g[0, 0, 0] = np.nan
    if kind == 'duplicate': inputs['source_ids'][1] = inputs['source_ids'][0]
    if kind == 'shape': g = g[:, :, :8]
    with pytest.raises(ValueError): analyze(inputs, g, a, b, cfg)


def test_config_and_protocol():
    cfg = read_settings(Path(__file__).resolve().parents[1]/'config/conditional_tau_weight_factor_swap.yaml')
    assert cfg['A']['run'] == 'zrv2yfgt'
    assert cfg['B']['run'] == 'n3jjqcn7'
    assert cfg['source_workers'] == 16
    a = {key: key for key in PROTOCOL_KEYS}
    b = dict(a, explicit_input=dict(arm='geometry'))
    verify_protocol(a, b)
    b['lr'] = 123
    with pytest.raises(ValueError, match='protocol differs: lr'): verify_protocol(a, b)


def test_execute_small_panel_with_real_outputs(tmp_path, monkeypatch):
    from scripts import diagnose_tau_weight_factor_swap as driver
    import evenet_dgpo.evenet.dataset.filtered_data as filtered
    inputs, g, a, b, cfg = fixture()
    # Exercise the entire report/NPZ/plot workflow without production files or W&B.
    expected, _ = analyze(inputs, g, a, b, cfg)
    previous = dict(truth=np.asarray(expected['truth_C']).reshape(9).tolist(), arms=[dict(
        arm='unweighted', cij=np.asarray(expected['arms']['unweighted']['C']).reshape(9).tolist(),
        error=expected['arms']['unweighted']['error'], event_ess=expected['arms']['unweighted']['event_ess'])])
    np.savez(tmp_path/'inputs.npz', visible_pt_sum=inputs['visible_pt_sum'])
    settings = dict(cfg, source=str(tmp_path), train_events='filtered_train', A='A', B='B')
    ma = {k: k for k in PROTOCOL_KEYS}; ma['condition_pt_edges'] = cfg['condition_pt_edges']
    mb = dict(ma, explicit_input=dict(arm='geometry'))
    monkeypatch.setattr(driver, 'load_source', lambda _: ({}, inputs, {'cij': g}, previous, {'rows': 60}))
    monkeypatch.setattr(filtered, 'validate_filtered_dataset', lambda _: {'rows': 416701})
    def model(spec, *_):
        name = 'AA' if spec == 'A' else 'BB'
        return (ma if spec == 'A' else mb, a if spec == 'A' else b,
                dict(truth_C=expected['truth_C'], arms=dict(bounded=expected['arms'][name])))
    monkeypatch.setattr(driver, 'read_model', model)
    result = driver.execute(settings, tmp_path)
    assert result['endpoints_verified'] is True
    assert (tmp_path/'COMPLETE').is_file()
    assert (tmp_path/'factor_swap.png').stat().st_size > 1000
    with np.load(tmp_path/'event_factors_and_bootstrap.npz') as stored:
        np.testing.assert_array_equal(stored['arm_names'], ARMS)
