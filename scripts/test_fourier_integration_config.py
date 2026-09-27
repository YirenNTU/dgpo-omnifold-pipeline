from pathlib import Path
from copy import deepcopy
import subprocess
import sys

import pytest

from train_fourier_integration import ARMS, PRETRAIN_ARMS, REPO_ROOT, make_config, validate_config


def test_three_arms_share_training_data_and_weights(tmp_path):
    configs = [make_config(a, tmp_path / "baseline.ckpt", tmp_path, 42) for a in ("none", "input", "output")]
    common = []
    for cfg in configs:
        validate_config(cfg)
        t = dict(cfg['options']['Training'])
        t.pop('model_checkpoint_save_path')
        common.append(t)
        assert cfg['experiment']['source_wandb_run'].endswith('/bvp5rn76')
    assert common[0] == common[1] == common[2]
    assert configs[0]['platform'] == configs[1]['platform'] == configs[2]['platform']
    branches = [c['network']['Body']['PET']['visible_angular_fourier'] for c in configs]
    assert not branches[0]['enabled']
    assert (branches[1]['placement'], branches[1]['projection_type']) == ('input', 'linear')
    assert (branches[2]['placement'], branches[2]['projection_type']) == ('output', 'mlp')
    assert len({c['options']['Training']['model_checkpoint_save_path'] for c in configs}) == 3
    assert len({c['logger']['wandb']['run_name'] for c in configs}) == 3


def test_pretrain_lr_arms_only_change_branch_learning_rate(tmp_path):
    assert PRETRAIN_ARMS == ('none', 'input', 'input_slow', 'input_fast')
    assert 'none' in ARMS
    reference = make_config('input', tmp_path / 'baseline.ckpt', tmp_path)
    for arm, rate in [('input', 2e-5), ('input_slow', 2e-6), ('input_fast', 6e-5)]:
        cfg = make_config(arm, tmp_path / 'baseline.ckpt', tmp_path)
        validate_config(cfg)
        assert cfg['network'] == reference['network']
        assert cfg['platform'] == reference['platform']
        candidate, expected = deepcopy(cfg['options']), deepcopy(reference['options'])
        for options in (candidate, expected):
            options['Training'].pop('model_checkpoint_save_path')
        branch = candidate['Training']['Components']['PET.angular_conditioning']
        assert branch['learning_rate'] == rate
        branch['learning_rate'] = 2e-5
        assert candidate == expected
    assert len({make_config(a, 'baseline.ckpt', tmp_path)['logger']['wandb']['run_name']
                for a in ARMS}) == len(ARMS)


def test_validation_rejects_accidental_full_resume(tmp_path):
    cfg = make_config('output', 'baseline.ckpt', tmp_path)
    cfg['options']['Training']['model_checkpoint_load_path'] = 'old.ckpt'
    with pytest.raises(ValueError, match='weights-only'):
        validate_config(cfg)


def test_input_mlp_preserves_source_training_and_injection_site(tmp_path):
    base = make_config('input', None, tmp_path)
    mlp = make_config('input_mlp', None, tmp_path)
    validate_config(mlp)
    assert mlp['platform'] == base['platform']
    for cfg in (base, mlp):
        cfg['options']['Training'].pop('model_checkpoint_save_path')
    assert mlp['options'] == base['options']
    branch = mlp['network']['Body']['PET']['visible_angular_fourier']
    assert branch['placement'] == 'input'
    assert branch['projection_type'] == 'mlp'
    assert branch.pop('mlp_dim') == 64
    branch['projection_type'] = 'linear'
    assert mlp['network'] == base['network']


def test_readout_pair_changes_only_frequency_bank(tmp_path):
    low = make_config('readout_k4', None, tmp_path)
    wide = make_config('readout_multiscale', None, tmp_path)
    baseline = make_config('input', None, tmp_path)
    for cfg in (low, wide, baseline):
        validate_config(cfg)
        cfg['options']['Training'].pop('model_checkpoint_save_path')
    assert low['options'] == wide['options'] == baseline['options']
    assert low['platform'] == wide['platform'] == baseline['platform']
    branch = wide['network']['Body']['PET']['visible_angular_fourier']
    assert branch['placement'] == 'output'
    assert branch['projection_type'] == 'cross_attention'
    assert branch['harmonics'] == [1, 2, 4, 8]
    branch['harmonics'] = [1, 2, 3, 4]
    assert low['network'] == wide['network']


def test_check_only_needs_no_checkpoint_and_does_not_launch(tmp_path):
    result = subprocess.run([sys.executable, str(REPO_ROOT / 'scripts/train_fourier_integration.py'),
        '--arm', 'output', '--output-root', str(tmp_path), '--check-only'],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'strict_no_fourier_source: true' in result.stdout
    assert 'epoch=190_train=0.1381_val=0.1263.ckpt' in result.stdout
    assert not list(tmp_path.iterdir())


def test_real_launch_requires_checkpoint(tmp_path):
    result = subprocess.run([sys.executable, str(REPO_ROOT / 'scripts/train_fourier_integration.py'),
        '--arm', 'input', '--checkpoint', str(tmp_path / 'missing.ckpt'),
        '--output-root', str(tmp_path)], capture_output=True, text=True)
    assert result.returncode != 0 and 'Missing source checkpoint' in result.stderr
    assert not list(tmp_path.iterdir())


def test_no_fourier_is_a_launchable_training_arm_with_yaml_checkpoint(tmp_path):
    result = subprocess.run([sys.executable, str(REPO_ROOT / 'scripts/train_fourier_integration.py'),
        '--arm', 'none', '--output-root', str(tmp_path), '--check-only'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    import yaml
    cfg = yaml.safe_load(result.stdout)
    assert not cfg['network']['Body']['PET']['visible_angular_fourier']['enabled']
    assert cfg['options']['Training']['pretrain_model_load_path'].endswith('epoch=190_train=0.1381_val=0.1263.ckpt')
    assert cfg['options']['Training']['freeze_modules'] == []
    assert not list(tmp_path.iterdir())
