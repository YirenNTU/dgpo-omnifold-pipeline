import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import yaml

from scripts.run_tau_cap_confirmation import (read_settings, stream_seeds, validate_replication,
                                              analyze, finish)
from scripts.run_tau_sampling_convergence import new_candidate_indices
from scripts.test_tau_tail_attribution import fixture
from scripts.tau_tail_attribution import analyze_prefix


def test_configuration_and_disjoint_all_rank_candidate_streams(tmp_path):
    cfg = read_settings(Path(__file__).resolve().parents[1]/'config/conditional_tau_cap_confirmation.yaml')
    old = dict(cfg, seed=930481)
    assert len(stream_seeds(cfg)) == 16*64
    assert not stream_seeds(cfg)&stream_seeds(old)
    assert cfg['seed'] == old['seed']+64*1000003
    assert list(new_candidate_indices(cfg)) == list(range(64))
    assert cfg['cap']==30 and cfg['prefixes']==[64]
    cfg['cap'] = 10
    file = tmp_path/'bad.yaml'; file.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='cap30'): read_settings(file)


def test_preflight_checks_same_inputs_protocol_and_no_rng_reuse(tmp_path):
    inputs, _, _ = fixture()
    source = tmp_path/'source'; source.mkdir()
    old = dict(classifier_run='pzq0nl1i', classifier_checkpoint='/head.pt', generator_checkpoint='/raw.ckpt',
        events='/filtered/val', conditions=len(inputs['weight']), runtime='/runtime.yaml', workers=16,
        batch_size=1024, feature_batch_size=256, ddim_steps=20, packing_spec={'shapes':{}},
        weights='raw_state_dict_only', ratio_transform='raw', historical_cij_errors=[.5, 1.4],
        condition_pt_edges=[1, 2, 3], seed=930481, prefixes=[1, 2, 4, 8, 16, 32, 64])
    (source/'manifest.json').write_text(json.dumps(old))
    (source/'wandb.json').write_text('{"id":"pt4n8f9t"}')
    (source/'COMPLETE').write_text('done')
    np.savez(source/'inputs.npz', **inputs)
    cfg = dict(old, source=str(source), source_run='pt4n8f9t', seed=64930673, prefixes=[64])
    out = validate_replication(cfg, inputs)
    assert out['inherited_candidates']==0 and out['seed_overlap_count']==0
    assert 'NOT independent-event' in out['evaluation_status']
    for seed in (930481, 930482):
        with pytest.raises(ValueError, match='overlap'): validate_replication(dict(cfg, seed=seed), inputs)
    with pytest.raises(ValueError, match='generator_checkpoint'):
        validate_replication(dict(cfg, generator_checkpoint='/ema.ckpt'), inputs)
    changed = dict(inputs, truth_cij=inputs['truth_cij']+1)
    with pytest.raises(ValueError, match='truth changed'): validate_replication(cfg, changed)


def test_only_three_arms_and_same_math_as_sensitivity_report():
    inputs, data, cfg = fixture()
    cfg.update(cap=30, ratio_caps=[30], temperatures=[])
    result = analyze(inputs, data, cfg)
    reference = analyze_prefix(inputs, data, 64, cfg, [.25, .5, .75])
    assert [row['arm'] for row in result['arms']]==['unweighted','raw','cap30']
    for row, old in zip(result['arms'], reference['arms']):
        for key in ('cij','error','event_ess','error_minus_unweighted_ci95'):
            np.testing.assert_allclose(row[key], old[key], atol=1e-12)
    assert result['independent_event_generalization_tested'] is False
    row = result['arms'][2]
    assert result['sampling_confirmation_passed'] == (row['error_minus_unweighted']<0 and row['error_minus_unweighted_ci95'][1]<0)


def test_identity_ratio_is_not_a_pass_and_matrices_have_paired_intervals():
    inputs, data, cfg = fixture()
    cfg['cap'] = 30; data['logits'][:] = 0
    result = analyze(inputs, data, cfg)
    assert not result['sampling_confirmation_passed']
    for row in result['arms']:
        np.testing.assert_allclose(row['error_minus_unweighted_ci95'], 0)
        assert np.array(row['residual_ci95']).shape == (2, 9)


def test_most_influential_exact_deletion():
    inputs, data, cfg = fixture()
    cfg['cap'] = 30; data['logits'][2, 30] = 15
    result = analyze(inputs, data, cfg)
    record = result['arms'][1]['most_influential']
    assert record['source_id']==inputs['source_ids'][2] and record['draw']==31
    raw = inputs['weight'][:, None]*np.exp(data['logits'])
    raw[2, 30] = 0; raw /= raw.sum()
    mean = np.einsum('nk,nkd->d', raw, data['cij'])
    assert record['leave_one_error']==pytest.approx(np.linalg.norm(mean-result['truth']))


def test_complete_report_logs_to_real_summary_without_generation(tmp_path):
    inputs, data, cfg = fixture()
    cfg.update(cap=30, conditions=len(inputs['weight']), output=str(tmp_path), prefixes=[64])
    np.savez(tmp_path/'inputs.npz', **inputs)
    from wandb.sdk.wandb_summary import Summary
    values = {}; summary = Summary(lambda: values)
    summary._set_update_callback(lambda record: values.update({x.key[0]: x.value for x in record.update}))
    logs, saved = [], []
    run = SimpleNamespace(summary=summary, log=logs.append, save=lambda *a, **k: saved.append(a), url='offline-test')
    fake = SimpleNamespace(Table=lambda **kw:kw, Image=lambda path:path)
    with patch('scripts.run_tau_cap_confirmation.merge', return_value=(data, .0001)), patch.dict('sys.modules', {'wandb':fake}):
        finish(cfg, run)
    report = json.loads((tmp_path/'cap_confirmation_report.json').read_text())
    assert values['phase']=='complete' and values['reused_samples']==0
    assert values['independent_event_generalization_tested'] is False
    assert report['candidates']==64 and len(report['arms'])==3
    assert report['classifier_fits']==report['policy_updates']==0
    assert (tmp_path/'Cij_confirmation.png').is_file() and (tmp_path/'COMPLETE').is_file()
    assert len(saved)==2
