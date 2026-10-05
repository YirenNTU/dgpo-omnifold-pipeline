import copy
import json
from pathlib import Path
import sys
from unittest import mock

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_dgpo_conditioning import ARMS, STEPS, build_arm_config, main, simultaneous_intervals, retry_incomplete_arm
from diagnose_dgpo_checkpoint_transfer import build_probe_config, source_metadata, DEFAULT_CONFIG
from test_dgpo_checkpoint_transfer import checkpoint
from train_neutrino_backend import REPO_ROOT


def source_runtime(tmp_path):
    meta = source_metadata(checkpoint())
    cfg = build_probe_config(REPO_ROOT / 'config/train_diffusion_nersc.yaml', DEFAULT_CONFIG,
        pinned=tmp_path / 'source.ckpt', output=tmp_path / 'old', metadata=meta)
    path = tmp_path / 'source_runtime.yaml'
    path.write_text(yaml.safe_dump(cfg))
    return path, cfg, meta


def test_three_arms_preserve_native_state_and_only_change_basis(tmp_path):
    path, old, meta = source_runtime(tmp_path)
    configs = {a: build_arm_config(path, pinned=tmp_path/'source.ckpt', output=tmp_path/a,
        root=tmp_path, metadata=meta, arm=a, run_id=f'test{a}') for a in ARMS}
    for a, cfg in configs.items():
        assert cfg['platform'] == old['platform']
        assert cfg['network']['Body'] == old['network']['Body']
        assert cfg['reward_config'] == old['reward_config']
        assert cfg['dgpo']['reference_trust'] == old['dgpo']['reference_trust']
        assert cfg['dgpo']['lr_schedule'] == old['dgpo']['lr_schedule']
        assert cfg['dgpo']['conditioning_learning_rates'] == old['dgpo']['conditioning_learning_rates']
        assert cfg['options']['Training']['Components'] == old['options']['Training']['Components']
        assert cfg['dgpo']['checkpoint_transfer']['relative_steps'] == STEPS
        assert cfg['dgpo']['gradient_transfer_trace']['update_end_steps'] == [1137+s for s in STEPS[1:]]
        assert cfg['experiment']['endpoint_selection'] == 'predeclared_relative_step50'
        assert cfg['logger']['wandb']['group'] == 'Late reward conditioning screen'
        assert len(cfg['logger']['wandb']['run_name']) <= 96
        assert not cfg['dgpo']['adaptive_omnifold']['recalibration']['refit_once_on_resume']
    assert configs['A']['network'] == old['network']
    b, c = (copy.deepcopy(configs[a]['network']) for a in ('B', 'C'))
    b['VisibleConditioning']['diffusion_reward_probe']['basis'] = 'relative'
    assert b == c


def test_prepare_does_not_start_compute_or_wandb(tmp_path, monkeypatch):
    path, _, _ = source_runtime(tmp_path)
    ckpt = tmp_path / 'last.ckpt'
    torch.save(checkpoint(), ckpt)
    monkeypatch.setattr(sys, 'argv', ['probe', '--source-runtime', str(path), '--checkpoint', str(ckpt),
                                    '--output-root', str(tmp_path/'output'), '--prepare-only'])
    with mock.patch('diagnose_dgpo_conditioning.subprocess.run') as run:
        main()
        run.assert_not_called()
    root, = (tmp_path/'output').iterdir()
    plan = json.loads((root/'plan.json').read_text())
    assert plan['status'] == 'prepared'
    assert plan['steps'][-1] == 50
    assert set(plan['arms']) == {'A', 'B', 'C'}
    assert len({v['run_id'] for v in plan['arms'].values()}) == 3


def test_simultaneous_intervals_are_paired_by_event():
    values = np.arange(100) / 100
    stats = simultaneous_intervals({'gain': values, 'delta': np.full(100, .02)}, replicates=100)
    assert stats['gain']['lo95_simultaneous'] < .495 < stats['gain']['hi95_simultaneous']
    assert stats['delta']['mean'] == pytest.approx(.02)
    assert stats['delta']['lo95_simultaneous'] == pytest.approx(.02)
    with pytest.raises(ValueError):
        simultaneous_intervals({'bad': np.array([np.nan, 1])})


def prepared_screen(tmp_path):
    path, _, meta = source_runtime(tmp_path)
    plan = {'source': meta, 'steps': STEPS, 'arms': {}}
    for arm in ARMS:
        output = tmp_path / arm
        output.mkdir()
        cfg = build_arm_config(path, pinned=tmp_path/'source.ckpt', output=output,
                               root=tmp_path, metadata=meta, arm=arm, run_id=f'test{arm}')
        runtime = output / 'runtime.yaml'
        runtime.write_text(yaml.safe_dump(cfg))
        plan['arms'][arm] = {'run_id': f'test{arm}', 'output': str(output), 'runtime': str(runtime)}
        if arm != 'C':
            (output / 'measurements').mkdir()
            (output / 'measurements' / 'report.json').write_text(json.dumps({
                'status': 'complete' if arm == 'A' else 'running', 'primary_endpoint': 50,
                'measurements': {'heldout/+0': {'delta_mean': 0}}}))
    (tmp_path / 'update_cache').mkdir()
    (tmp_path / 'update_cache' / 'update001_rank00.pt').write_bytes(b'cached A input')
    (tmp_path / 'plan.json').write_text(json.dumps(plan))
    return plan


def test_retry_preserves_A_cache_failed_evidence_and_source_configuration(tmp_path):
    plan = prepared_screen(tmp_path)
    before = copy.deepcopy(plan)
    old_cfg = yaml.safe_load((tmp_path/'B/runtime.yaml').read_text())
    old_report = (tmp_path/'B/measurements/report.json').read_bytes()
    a_report = (tmp_path/'A/measurements/report.json').read_bytes()
    retry_incomplete_arm(tmp_path, plan, 'B')
    saved = json.loads((tmp_path/'plan.json').read_text())
    assert plan == saved
    assert plan['arms']['A'] == before['arms']['A']
    assert plan['arms']['C'] == before['arms']['C']
    assert plan['arms']['B']['run_id'] != 'testB'
    archive = Path(plan['interrupted_attempts'][0]['output'])
    assert (archive/'measurements/report.json').read_bytes() == old_report
    assert (tmp_path/'A/measurements/report.json').read_bytes() == a_report
    assert (tmp_path/'update_cache/update001_rank00.pt').read_bytes() == b'cached A input'
    assert not (tmp_path/'B/measurements').exists()
    cfg = yaml.safe_load((tmp_path/'B/runtime.yaml').read_text())
    assert cfg['logger']['wandb']['id'] == plan['arms']['B']['run_id']
    assert cfg['logger']['wandb']['resume'] == 'never'
    assert len(cfg['logger']['wandb']['run_name']) <= 96
    assert cfg['experiment']['retry_of_run'] == 'testB'
    for key in old_cfg:
        if key not in ('logger', 'experiment'):
            assert cfg[key] == old_cfg[key]


def test_retry_launch_skips_completed_A_and_runs_B_then_C(tmp_path, monkeypatch):
    prepared_screen(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['probe', '--prepared', str(tmp_path), '--retry-arm', 'B'])
    with mock.patch('diagnose_dgpo_conditioning.subprocess.run') as run, \
         mock.patch('diagnose_dgpo_conditioning.summarize') as summary:
        main()
    commands = [c.args[0] for c in run.call_args_list]
    assert len(commands) == 2
    assert str(tmp_path/'B/runtime.yaml') in commands[0]
    assert str(tmp_path/'C/runtime.yaml') in commands[1]
    assert all(str(tmp_path/'A/runtime.yaml') not in c for c in commands)
    assert all(c[c.index('--max-steps')+1] == '1187' for c in commands)
    summary.assert_called_once()


def test_retry_rejects_completed_B_without_touching_it(tmp_path):
    plan = prepared_screen(tmp_path)
    report = tmp_path/'B/measurements/report.json'
    report.write_text(json.dumps({'status': 'complete', 'primary_endpoint': 50}))
    original_plan = (tmp_path/'plan.json').read_bytes()
    with pytest.raises(ValueError, match='complete'):
        retry_incomplete_arm(tmp_path, plan, 'B')
    assert report.exists()
    assert (tmp_path/'plan.json').read_bytes() == original_plan
    assert not list(tmp_path.glob('B.interrupted-*'))


def test_retry_requires_completed_A(tmp_path):
    plan = prepared_screen(tmp_path)
    (tmp_path/'A/measurements/report.json').write_text(json.dumps({'status': 'running'}))
    with pytest.raises(ValueError, match='completed 50-update A'):
        retry_incomplete_arm(tmp_path, plan, 'B')
    assert (tmp_path/'B/measurements/report.json').exists()
