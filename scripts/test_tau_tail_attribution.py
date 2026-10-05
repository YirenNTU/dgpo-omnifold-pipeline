import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import yaml

from scripts.tau_tail_attribution import (candidate_weights, attribution, joint_bins,
    clustered_bin_comparison, analyze_prefix, paired_bootstrap)
from scripts.diagnose_tau_sampling_tail import load_source, read_settings, verify_raw, execute
from scripts.tau_sampling_convergence import convergence_report


def fixture(n=36, k=64):
    rng = np.random.default_rng(824)
    directions = rng.normal(size=(n, k, 2, 3))
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    a, b = directions[..., 0, :], directions[..., 1, :]
    tau = np.concatenate((a, b, np.einsum('nki,nkj->nkij', a, b).reshape(n, k, 9)), -1)
    cij = rng.normal(size=(n, k, 9))
    data = dict(logits=rng.normal(size=(n, k)), cij=cij, tau=tau, deltas=np.zeros((n, k, 2, 2)))
    inputs = dict(weight=rng.uniform(.5, 2, n), source_ids=np.arange(n).astype(str),
        truth_cij=rng.normal(size=(n, 9)), truth_tau=tau[:, 0], category=np.full(n, 11),
        visible_pt_sum=np.arange(n)/n, kappas=np.ones((n, 2)), visible_a=np.ones((n, 4)), visible_b=np.ones((n, 4)))
    cfg = dict(ratio_caps=[2, 10], temperatures=[.5], top_count=3, remove_counts=[1, 4],
        bootstrap=20, bootstrap_seed=42, minimum_truth_bin_count=2)
    return inputs, data, cfg


def test_weights_match_direct_global_formula_and_ignore_zero_base():
    base = np.array([1., 3., 0.])
    scores = np.array([[2., -1.], [3., 0.], [99999., 99999.]])
    w, log_mean = candidate_weights(base, scores)
    direct = base[:2, None]*np.exp(scores[:2])
    np.testing.assert_allclose(w[:2], direct/direct.sum())
    np.testing.assert_array_equal(w[2], 0)
    assert log_mean == pytest.approx(np.log(direct.sum()/(base.sum()*2)))
    shifted, lm = candidate_weights(base, scores+5000)
    np.testing.assert_allclose(shifted, w)
    assert lm == pytest.approx(log_mean+5000)
    with pytest.raises(ValueError): candidate_weights(-base, scores)


def test_leave_one_and_multiple_removal_are_exact_full_truth_unchanged():
    inputs, data, cfg = fixture(k=8)
    data['logits'][3, 5] = 10
    report = attribution(inputs, data, 8, cfg)
    w, _ = candidate_weights(inputs['weight'], data['logits'])
    mean = np.einsum('nk,nkd->d', w, data['cij'])
    for record in report['top_candidates']:
        ww = w.copy(); ww[record['event_index'], record['draw']-1] = 0; ww /= ww.sum()
        revised = np.einsum('nk,nkd->d', ww, data['cij'])
        np.testing.assert_allclose(record['exact_leave_one_change'], revised-mean, atol=1e-12)
    selected = np.argsort(-w.ravel())[:4]
    ww = w.copy().ravel(); ww[selected] = 0; ww /= ww.sum()
    expected = np.einsum('n,nd->d', ww, data['cij'].reshape(-1, 9))
    row = next(r for r in report['removal_sensitivity'] if r['ranking']=='mass' and r['removed']==4)
    np.testing.assert_allclose(row['cij'], expected)
    assert row['error'] == pytest.approx(np.linalg.norm(expected-np.average(inputs['truth_cij'], weights=inputs['weight'], axis=0)))
    np.testing.assert_allclose(np.sum([b['cij_contribution'] for b in report['draw_blocks']], axis=0), mean)
    np.testing.assert_allclose(np.sum([b['centered_contribution'] for b in report['draw_blocks']], axis=0), 0, atol=1e-12)


def test_histogram_cluster_error_not_shrunk_by_duplicating_candidates():
    tb = np.array([0, 0, 1, 1, 0, 1]); gb = np.array([0, 1, 0, 1, 0, 1])[:, None]
    base = np.array([1., 2., 1., 2., 1., 1.])
    w, _ = candidate_weights(base, np.zeros((6, 1)))
    a = clustered_bin_comparison(tb, gb, base, w, 2)
    b = clustered_bin_comparison(tb, np.repeat(gb, 64, 1), base, np.repeat(w/64, 64, 1), 2)
    for i in range(4): np.testing.assert_allclose(a[i], b[i], atol=1e-12)
    exact = clustered_bin_comparison(tb, tb[:, None], base, w, 2)
    np.testing.assert_allclose(exact[2], 0, atol=1e-12)


def test_phi_periodic_boundary_and_fixed_bins():
    tau = np.zeros((2, 15)); tau[:, 0] = -1; tau[:, 3] = 1
    tau[0, 1], tau[1, 1] = 0., -0.
    results = joint_bins(tau)
    assert results['joint_phi'][0][0] == results['joint_phi'][0][1]
    for cells, count in results.values(): assert ((cells >= 0)&(cells < count)).all()


def test_raw_replay_and_paired_bootstrap_match_existing_report():
    inputs, data, cfg = fixture(k=8)
    out = analyze_prefix(inputs, data, 8, cfg, [.25, .5, .75])
    old = convergence_report(inputs['truth_cij'], data['cij'], data['logits'], inputs['weight'], bootstrap=20, seed=42)
    verify_raw(out, old['prefixes']['8'])
    raw = next(r for r in out['arms'] if r['arm']=='raw')
    np.testing.assert_allclose(raw['error_minus_unweighted_ci95'], old['prefixes']['8']['error_change_ci95'], atol=1e-10)
    for r in out['arms']:
        assert r['sensitivity_only'] == (r['arm'] not in ('raw', 'unweighted'))
    assert all(r['coarse_raw_context']['joint_phi'] is not None for r in out['attribution']['top_candidates'])
    # Each partition reproduces the stratum mass; no per-event ratio normalization.
    hist = out['closure']['raw']
    for group in {r['group'] for r in hist}:
        rows = [r for r in hist if r['group']==group and r['family']=='joint_phi']
        assert sum(r['generated_probability'] for r in rows) == pytest.approx(1)
        assert sum(r['generated_global_mass'] for r in rows) == pytest.approx(rows[0]['weighted_group_mass'])
    old['prefixes']['8']['reweighted'][0] += .1
    with pytest.raises(ValueError, match='reproduce'): verify_raw(out, old['prefixes']['8'])


def test_paired_bootstrap_keeps_truth_candidates_in_one_event():
    inputs, _, cfg = fixture()
    base = inputs['weight']/inputs['weight'].sum()
    num = base[:, None]*inputs['truth_cij']
    estimates, targets = paired_bootstrap(inputs, [num, num], [base, base], 20, 42)
    np.testing.assert_allclose(estimates[:, 0], targets, atol=1e-12)
    np.testing.assert_array_equal(estimates[:, 0], estimates[:, 1])


def test_cap_and_tempering_match_explicit_transforms_not_event_normalization():
    inputs, data, cfg = fixture(k=8)
    data['logits'][2, 3] = 12
    result = analyze_prefix(inputs, data, 8, cfg, [.25, .5, .75])
    for name, logits in [('cap_2', np.minimum(data['logits'], np.log(2))),
                         ('alpha_0.5', .5*data['logits'])]:
        row = next(r for r in result['arms'] if r['arm']==name)
        w = inputs['weight'][:, None]*np.exp(logits)
        w /= w.sum()
        expected = np.einsum('nk,nkd->d', w, data['cij'])
        np.testing.assert_allclose(row['cij'], expected, atol=1e-12)
        assert row['sensitivity_only']
    # Unit-ratio baseline has nonzero support bin ESS and no spurious mass drift.
    for row in result['closure']['unweighted']:
        assert row['base_group_mass'] == pytest.approx(row['weighted_group_mass'])
        if row['unweighted_global_mass'] > 0:
            assert row['applied_coarse_ratio'] == pytest.approx(1)


def test_unsupported_bins_are_not_reported_as_measured_ratios():
    inputs, data, cfg = fixture(k=8)
    # Put every generated direction in the same angular cell.
    data['tau'][:] = data['tau'][0, 0]
    result = analyze_prefix(inputs, data, 8, cfg, [.25, .5, .75])
    empty = [r for r in result['closure']['raw'] if r['unweighted_global_mass']==0]
    assert empty
    assert all(r['required_coarse_ratio'] is None and r['applied_coarse_ratio'] is None
               and r['weighted_bin_event_ess']==0 and r['sparse'] for r in empty)


def source_fixture(tmp_path):
    inputs, data, cfg = fixture()
    source = tmp_path/'source'; source.mkdir()
    manifest = dict(weights='raw_state_dict_only', classifier_run='pzq0nl1i', workers=16,
                    conditions=len(inputs['weight']), prefixes=[1, 2, 4, 8, 16, 32, 64],
                    classifier_fits=0, policy_updates=0, condition_pt_edges=[.25, .5, .75])
    old = convergence_report(inputs['truth_cij'], data['cij'], data['logits'], inputs['weight'],
                             prefixes=manifest['prefixes'], bootstrap=20)
    old['extension'] = dict(old_prefixes_reproduced=True)
    (source/'manifest.json').write_text(json.dumps(manifest))
    (source/'wandb.json').write_text(json.dumps(dict(id='pt4n8f9t')))
    (source/'convergence_report.json').write_text(json.dumps(old))
    (source/'COMPLETE').write_text('done')
    np.savez(source/'inputs.npz', **inputs)
    np.savez(source/'samples_and_scores.npz', source_ids=inputs['source_ids'], **data)
    cfg.update(source=str(source), source_run='pt4n8f9t', expected_events=len(inputs['weight']),
               expected_candidates=64, prefixes=[32, 64])
    return cfg, inputs, data


def test_source_identity_and_protocol_checks(tmp_path):
    cfg, inputs, data = source_fixture(tmp_path)
    load_source(cfg)
    with pytest.raises(ValueError, match='Source run'): load_source(dict(cfg, source_run='wrong'))
    np.savez(Path(cfg['source'])/'samples_and_scores.npz', source_ids=inputs['source_ids'][::-1], **data)
    with pytest.raises(ValueError, match='identities'): load_source(cfg)


def test_complete_offline_execution_writes_reports_source_unchanged(tmp_path):
    cfg, _, _ = source_fixture(tmp_path)
    source = Path(cfg['source'])
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    output = tmp_path/'output'; output.mkdir()
    # Real W&B Summary catches the previously encountered update(**kwargs) API error.
    from wandb.sdk.wandb_summary import Summary
    values = {}
    summary = Summary(lambda: values)
    summary._set_update_callback(lambda record: values.update({x.key[0]: x.value for x in record.update}))
    logs, saves = [], []
    run = SimpleNamespace(summary=summary, log=lambda x: logs.append(x), save=lambda *a, **k: saves.append(a))
    fake = SimpleNamespace(Table=lambda **kw: kw, Image=lambda path: path)
    with patch.dict('sys.modules', {'wandb': fake}):
        report = execute(cfg, output, run)
    assert values['phase']=='complete' and values['original_endpoints_verified']
    assert report['generated_samples']==report['classifier_fits']==report['policy_updates']==0
    assert (output/'COMPLETE').is_file() and (output/'sensitivity.png').is_file()
    assert all((source/name).read_bytes()==content for name, content in before.items())
    assert set(p.name for p in source.iterdir())==set(before)
    assert len(saves)==4 and len(logs)>10
    saved = json.loads((output/'tail_attribution_report.json').read_text())
    assert len(saved['prefixes']['64']['arms'])==5


def test_config_pins_current_panel_and_grid(tmp_path):
    path = Path(__file__).resolve().parents[1]/'config/conditional_tau_tail_attribution.yaml'
    cfg = read_settings(path)
    assert cfg['source_run']=='pt4n8f9t' and cfg['expected_events']==119002
    assert cfg['prefixes']==[32, 64] and cfg['cpu_threads']==16
    cfg['temperatures'] = [float('nan')]
    bad = tmp_path/'bad.yaml'; bad.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='grid'): read_settings(bad)
