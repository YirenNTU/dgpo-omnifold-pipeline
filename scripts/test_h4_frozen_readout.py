import copy
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_h4_frozen_readout as capture
import replay_h4_frozen_readout as replay
from RL.DGPO_neutrino.omnifold_ztautau.stage import build_fit_config


def test_capture_contract():
    cfg = capture.validated_config()
    fit = build_fit_config(cfg['dgpo']['adaptive_omnifold']['audit_fit'], n_train=20000, n_validation=4000)
    fit.validate()
    assert fit.representation_export_dir == capture.OUTPUT + '/panels'
    assert fit.learning_rate == 2e-4 and fit.lr_scheduler == 'constant'
    assert cfg['platform']['number_of_workers'] == 16
    assert cfg['logger']['wandb']['resume'] == 'never'
    assert len(cfg['logger']['wandb']['run_name']) < 96 and len(replay.NAME) < 96


def test_quick_capture_contract():
    cfg = capture.validated_config(True)
    fit = build_fit_config(cfg['dgpo']['adaptive_omnifold']['audit_fit'], n_train=20000, n_validation=4000)
    fit.validate()
    assert fit.steps == fit.min_steps == 300
    assert not fit.require_saturation
    assert cfg['logger']['wandb']['id'] == 'h4frzq1'
    assert cfg['platform']['number_of_workers'] == 16


def test_reject_architecture_change():
    cfg = capture.validated_config()
    cfg['dgpo']['adaptive_omnifold']['audit_fit']['learning_rate'] = .1
    with patch.object(capture, 'read_overlay_yaml', return_value=cfg):
        with pytest.raises(ValueError, match='preserve'):
            capture.validated_config()


def data():
    g = torch.Generator().manual_seed(88)
    y = torch.arange(256) % 2 == 0
    x = torch.randn(256, 4, generator=g)
    x[:, 0] += y.float() * 5 - 2.5
    fit = torch.arange(256) < 128
    return x, y, fit


def test_linear_learning_and_reproducible_frozen_features():
    torch.set_num_threads(1)
    x, y, fit = data()
    original = x.clone()
    state = torch.get_rng_state().clone()
    a = replay.fit_head(x, y, fit, standardized=True, seed=17, steps=100, lr=.02, batch_size=32)
    b = replay.fit_head(x, y, fit, standardized=True, seed=17, steps=100, lr=.02, batch_size=32)
    assert a == b
    assert torch.equal(x, original) and x.grad is None
    assert torch.equal(state, torch.get_rng_state())
    assert a['curve'][0]['holdout_auc'] == .5
    assert a['curve'][-1]['holdout_auc'] > .98
    assert a['curve'][-1]['holdout_bce'] < .3


def test_holdout_never_affects_fit_or_standardizer():
    x, y, fit = data()
    kwargs = dict(standardized=True, seed=29, steps=25, batch_size=32)
    a = replay.fit_head(x, y, fit, **kwargs)
    x[~fit] *= -100
    y[~fit] = ~y[~fit]
    b = replay.fit_head(x, y, fit, **kwargs)
    for key in ('weights', 'bias', 'mean', 'scale'):
        assert a[key] == b[key]


def test_full_batch_seed_independent_and_no_holdout_leakage():
    x, y, fit = data()
    kwargs = dict(standardized=True, full_batch=True, steps=50, lr=.02)
    a = replay.fit_head(x, y, fit, seed=17, **kwargs)
    b = replay.fit_head(x, y, fit, seed=43, **kwargs)
    assert a == b
    assert a['curve'][-1]['holdout_auc'] > .98
    assert a['curve'][-1]['rows_processed'] == 50 * int(fit.sum())
    x[~fit] *= 100
    y[~fit] = ~y[~fit]
    c = replay.fit_head(x, y, fit, seed=17, **kwargs)
    for key in ('weights', 'bias', 'mean', 'scale'):
        assert a[key] == c[key]


def test_batch_decision_requires_late_window_not_best_point():
    good = dict(curve=[dict(step=s, holdout_auc=.84, holdout_bce=.69) for s in (200,225,250,275,300)])
    arms = {'full_batch': copy.deepcopy(good), 'minibatch/seed17': copy.deepcopy(good)}
    arms['minibatch/seed17']['curve'][-1]['holdout_auc'] = .57
    assert 'supports minibatch' in replay.batch_decision(arms, .842, .502)['decision']
    arms['full_batch']['curve'][-1]['holdout_auc'] = .7
    assert 'does not suffice' in replay.batch_decision(arms, .842, .502)['decision']


def test_decision_does_not_use_best_holdout_checkpoint():
    arms = {f'{kind}/seed{s}': {'curve': [dict(step=200, holdout_auc=.79, holdout_bce=.6)]}
            for kind in ('raw', 'standardized') for s in (17, 29, 43)}
    assert replay.decide(arms, .8, .5).startswith('supports accessible')
    for key in arms:
        if key.startswith('raw/'):
            arms[key]['curve'][0]['holdout_auc'] = .6
    assert 'conditioning' in replay.decide(arms, .8, .5)
    assert 'inconclusive' in replay.decide(arms, .55, .5)


def test_bad_panel_fails():
    x, y, fit = data()
    panel = dict(schema=1, capture_step=50, target=y, fit_mask=fit,
                 inner_mask=torch.arange(256) % 4 < 2,
                 features=dict(raw_fourier=x, fourier=x), current_logits=torch.zeros(256))
    replay.validate_panel(panel)
    bad = copy.deepcopy(panel)
    bad['features']['fourier'][0, 0] = float('nan')
    with pytest.raises(ValueError, match='features'):
        replay.validate_panel(bad)


@pytest.mark.parametrize('batch_ablation', [False, True])
def test_complete_offline_replay(tmp_path, monkeypatch, batch_ablation):
    x, y, fit = data()
    panels = tmp_path / 'panels'
    panels.mkdir()
    for step in (0, 50, 100):
        torch.save(dict(schema=1, capture_step=step, world_size=16, target=y, fit_mask=fit,
                        inner_mask=torch.arange(256) % 4 < 2, fit_config={},
                        features=dict(raw_fourier=x, fourier=x), current_logits=torch.zeros(256)),
                   panels / f'panel_step{step:04d}.pt')
    monkeypatch.setitem(replay.SETTINGS, 'steps', 2)
    monkeypatch.setitem(replay.SETTINGS, 'eval_every', 1)
    monkeypatch.setitem(replay.SETTINGS, 'primary_update', 1)
    output = tmp_path / 'report'
    monkeypatch.setattr(sys, 'argv', ['replay', '--offline', '--panels', str(panels), '--output', str(output)]
                        + (['--batch-ablation'] if batch_ablation else []))
    replay.main()
    import json
    result = json.loads((output / 'report.json').read_text())
    assert set(result['captures']) == ({'50'} if batch_ablation else {'0', '50', '100'})
    assert result['historical_context']['h4clf02']['first_val_auc_085_update'] == 1280
    assert len(result['captures']['50']['branches']['fourier']['arms']) == (4 if batch_ablation else 6)
    if batch_ablation:
        assert result['settings']['steps'] == 300
        assert 'full_batch' in result['batch_comparison']['arms']
    with pytest.raises(FileExistsError):
        replay.main()
