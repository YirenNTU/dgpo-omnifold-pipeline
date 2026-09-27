import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_h4_classifier_representation as launcher
from RL.DGPO_neutrino.omnifold_ztautau.stage import build_fit_config


def test_contract():
    cfg = launcher.validated_config()
    fit = build_fit_config(cfg['dgpo']['adaptive_omnifold']['audit_fit'], n_train=20000, n_validation=4000)
    fit.validate()
    assert fit.representation_diagnostic_enabled and fit.lr_scheduler == 'constant'
    assert fit.learning_rate == 2e-4 and fit.adapter_learning_rate == fit.decoder_learning_rate == 5e-5
    assert fit.backbone_learning_rate == 1e-5 and fit.steps is None and fit.min_steps == 1000
    assert cfg['platform']['number_of_workers'] == 16
    assert cfg['experiment']['source_policy_step'] == 1110
    assert cfg['options']['Training']['model_checkpoint_load_path'] == launcher.SOURCE
    assert cfg['dgpo']['checkpoint_load_mode'] == 'weights_only'


def test_path_contract():
    cfg = launcher.validated_config(True)
    fit = build_fit_config(cfg['dgpo']['adaptive_omnifold']['audit_fit'], n_train=20000, n_validation=4000)
    fit.validate()
    assert fit.representation_path_enabled
    assert cfg['logger']['wandb']['id'] == 'h4clfpath1'
    assert cfg['logger']['wandb']['resume'] == 'never'
    assert cfg['options']['Training']['model_checkpoint_load_path'] == launcher.SOURCE


@pytest.mark.parametrize('section,key,value', [
    ('experiment', 'source_policy_step', 50),
    ('experiment', 'source_wandb_run', 'h4lbnr01'),
    ('provenance', 'source_dgpo_global_step', 50),
    ('provenance', 'source_checkpoint', 'wrong.ckpt'),
    ('training', 'model_checkpoint_load_path', 'wrong.ckpt'),
])
def test_reject_wrong_source(section, key, value):
    cfg = launcher.validated_config()
    target = {'experiment': cfg['experiment'],
              'provenance': cfg['nersc']['reproducibility'],
              'training': cfg['options']['Training']}[section]
    target[key] = value
    with patch.object(launcher, 'read_overlay_yaml', return_value=cfg):
        with pytest.raises(ValueError, match='1110'):
            launcher.validated_config()


def test_reject_lr_change():
    cfg = launcher.validated_config()
    cfg['dgpo']['adaptive_omnifold']['audit_fit']['learning_rate'] = .01
    with patch.object(launcher, 'read_overlay_yaml', return_value=cfg):
        with pytest.raises(ValueError, match='measurement'):
            launcher.validated_config()


def test_manifest(tmp_path):
    path = tmp_path / 'filter_manifest.json'
    manifest = dict(complete=True, rows_in=119004, rows_out=119002, events_removed=2,
                    output=launcher.CLEAN, removed_events=[{'source_event_key': k} for k in ('9628165', '3333501')])
    path.write_text(json.dumps(manifest))
    launcher.validate_manifest(path)
    manifest['rows_out'] = 119004
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='manifest'):
        launcher.validate_manifest(path)
