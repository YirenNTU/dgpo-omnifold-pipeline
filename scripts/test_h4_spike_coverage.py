import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_h4_spike_coverage import coverage, select_panel, angles, load_artifact, audit_ddim_steps
from diagnose_h4_ratio_tail import unpack
from test_h4_topology_resolution import fixture


def test_joint_and_any_hit_are_not_draw_fraction():
    t = np.zeros((2, 2))
    g = np.ones((2, 8, 2))
    g[0, 0] = 0
    g[1, 0, 0] = 0
    r = coverage(t, g)['regions']
    joint = next(x for x in r if x['region'] == 'joint')
    assert joint['truth_count'] == 2
    assert joint['generated_draw_fraction'] == 1/16
    assert joint['any_hit_event_fraction'] == .5
    assert joint['hit_fraction_given_truth_spike'] == .5
    azimuth = next(x for x in r if x['region'] == 'acoplanarity')
    assert azimuth['any_hit_event_fraction'] == 1
    json.dumps(r, allow_nan=False)


def test_empty_truth_region_and_zero_generated_hits():
    r = coverage(np.ones((2, 2)), np.ones((2, 3, 2)))
    assert all(x['hit_fraction_given_truth_spike'] is None for x in r['regions'])
    assert all(x['generated_draw_count'] == 0 for x in r['regions'])


def test_nested_prefixes():
    t = np.zeros((2, 2))
    g = np.ones((2, 128, 2))
    g[0, 7] = 0
    g[1, 31] = 0
    hits = [coverage(t, g[:, :k])['regions'][2]['any_hit_event_count'] for k in [1, 8, 32, 128]]
    assert hits == [0, 1, 2, 2]


def test_selection_reproducible_and_unique():
    a = select_panel(100, 32, 42)
    assert torch.equal(a, select_panel(100, 32, 42))
    assert len(a.unique()) == 32
    with pytest.raises(ValueError):
        select_panel(10, 11, 42)


def test_threshold_is_strict():
    r = coverage(np.full((2, 2), 1e-6), np.full((2, 1, 2), 1e-6))
    assert r['regions'][0]['truth_count'] == 0
    assert r['regions'][3]['truth_count'] == 2


def test_artifact_guard_and_reconstruction(tmp_path):
    b = fixture()
    torch.save(b, tmp_path / 'best_classifier_and_test.pt')
    with pytest.raises(ValueError, match='Incomplete'):
        load_artifact(tmp_path)
    (tmp_path / 'COMPLETE').write_text('h4-ratio-health-v1')
    loaded = load_artifact(tmp_path)
    assert angles(unpack(loaded), loaded['test_truth']).shape == (100, 2)
    b['truth_topology'][:, 0] = 100
    torch.save(b, tmp_path / 'best_classifier_and_test.pt')
    with pytest.raises(ValueError, match='mismatch'):
        load_artifact(tmp_path)


def test_bad_angles_fail():
    with pytest.raises(ValueError):
        coverage(np.zeros((2, 2)), np.full((2, 1, 2), np.nan))


def test_validation_sampler_budget():
    assert audit_ddim_steps({'dgpo': {'num_ddim_steps': 5, 'validation_num_ddim_steps': 20}}) == 20
    assert audit_ddim_steps({'dgpo': {'num_ddim_steps': 5, 'validation_num_ddim_steps': None}}) == 5
    assert audit_ddim_steps({'dgpo': {'num_ddim_steps': 5}}) == 5


def test_offline_cli(tmp_path):
    import subprocess
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'COMPLETE').write_text('h4-ratio-health-v1')
    torch.save(fixture(), source / 'best_classifier_and_test.pt')
    output = tmp_path / 'output'
    command = [sys.executable, str(Path(__file__).with_name('diagnose_h4_spike_coverage.py')),
        str(source), '--output', str(output), '--events', '32']
    subprocess.run(command, check=True, capture_output=True)
    assert json.loads((output / 'artifact_coverage.json').read_text())['events'] == 100
    assert not (output / 'COMPLETE').exists()  # no sampling claim
    assert subprocess.run(command, capture_output=True).returncode != 0


def test_raw_only_ignores_ema():
    from diagnose_h4_spike_coverage import load_raw_state
    model = torch.nn.Linear(2, 1)
    raw = {f'model.{k}': torch.ones_like(v) for k, v in model.state_dict().items()}
    source = dict(state_dict=raw, ema_state_dict={'weight': torch.full((1, 2), 999.)})
    load_raw_state(model, source, 'pretrained10pct')
    assert all(torch.equal(v, torch.ones_like(v)) for v in model.state_dict().values())
    with pytest.raises(ValueError, match='1110'):
        load_raw_state(model, source, 'step1110')
    source['global_step'] = 1110
    load_raw_state(model, source, 'step1110')
    source['dgpo_checkpoint_version'] = 1
    with pytest.raises(ValueError, match='DGPO'):
        load_raw_state(model, source, 'pretrained10pct')


def test_raw_incompatibility_fails():
    from diagnose_h4_spike_coverage import load_raw_state
    model = torch.nn.Linear(2, 1)
    with pytest.raises(ValueError, match='keys'):
        load_raw_state(model, {'state_dict': {}}, 'pretrained10pct')
    with pytest.raises(ValueError, match='shape/dtype'):
        load_raw_state(model, {'state_dict': {k: v.double() for k, v in model.state_dict().items()}}, 'pretrained10pct')


def test_matched_panel_and_settings(tmp_path):
    from argparse import Namespace
    from diagnose_h4_spike_coverage import matched_source
    b = fixture()
    ids = select_panel(100, 32, 42)
    args = Namespace(workers=16, events=32, batch_size=16, seed=42)
    manifest = dict(vars(args), K=128, policy_updates=0)
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    (tmp_path / 'COMPLETE').write_text('h4-spike-coverage-v1')
    panel = dict(test_rows=ids, pool_rows=b['split_indices']['test'][ids],
        truth=b['test_truth'][ids], condition=b['test_condition'][ids], packing_spec=b['packing_spec'])
    torch.save(panel, tmp_path / 'panel.pt')
    matched_source(tmp_path, args, b, ids)
    args.seed = 43
    with pytest.raises(ValueError, match='seed'):
        matched_source(tmp_path, args, b, ids)
    args.seed = 42
    panel['truth'] = panel['truth'] + 1
    torch.save(panel, tmp_path / 'panel.pt')
    with pytest.raises(ValueError, match='truth'):
        matched_source(tmp_path, args, b, ids)


def test_full_pretrain_raw_contract():
    from diagnose_h4_spike_coverage import load_raw_state, ARM_METADATA, FULL_PRETRAIN_CHECKPOINT
    model = torch.nn.Linear(2, 1)
    source = dict(state_dict={k: torch.ones_like(v) for k, v in model.state_dict().items()},
                  ema_state_dict={'weight': torch.full((1, 2), 999.)})
    load_raw_state(model, source, 'pretrainedfull')
    assert torch.equal(model.weight, torch.ones_like(model.weight))
    source['dgpo_checkpoint_version'] = 1
    with pytest.raises(ValueError, match='DGPO'):
        load_raw_state(model, source, 'pretrainedfull')
    assert FULL_PRETRAIN_CHECKPOINT.endswith('/diffusion_pretrain_v1/checkpoints/last.ckpt')
    assert ARM_METADATA['pretrainedfull'][0] == 'h4covfull1'
    assert all(len(name) < 96 for _, name in ARM_METADATA.values())


def test_paired_joint_comparison_direction():
    from diagnose_h4_spike_coverage import paired_joint_comparison
    before = np.ones((2, 128, 2))
    after = before.copy()
    after[0] = 0
    rows = paired_joint_comparison(before, after)
    assert len(rows) == 12
    assert all(row['current_minus_baseline_joint_fraction'] == .5 for row in rows)
    assert all(row['paired_event_se'] == pytest.approx(.5) for row in rows)
    assert all(row['current_minus_baseline_joint_fraction'] == 0 for row in paired_joint_comparison(before, before))
    with pytest.raises(ValueError, match='shape'):
        paired_joint_comparison(before[:, :8], after)
