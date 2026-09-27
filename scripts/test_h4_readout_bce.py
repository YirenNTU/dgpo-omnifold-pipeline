import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagnose_h4_readout_bce as diagnostic
from replay_h4_frozen_readout import fit_head


def test_balanced_attribution_and_ratio_tails():
    y = torch.tensor([True, False, False, False])
    a = torch.tensor([2., 1., 1., 1.])
    b = torch.tensor([1., 1., 1., 1.])
    result = diagnostic.attribution(a, b, y)
    assert result['balanced_bce_gain'] == .5
    assert result['balanced_fraction_improved'] == .5
    assert result['top1pct_positive_gain_share'] == 1
    null = diagnostic.ratio_metrics(torch.zeros(100))
    assert abs(null['ess_fraction'] - 1) < 1e-10
    extreme = diagnostic.ratio_metrics(torch.tensor([10000., -10000.]))
    assert extreme['ess_fraction'] == .5


def test_snapshot_callback_does_not_change_training():
    torch.set_num_threads(1)
    g = torch.Generator().manual_seed(4)
    x = torch.randn(256, 4, generator=g)
    y = torch.arange(256) % 2 == 0
    fit = torch.arange(256) < 128
    kwargs = dict(standardized=True, seed=17, steps=50)
    a = fit_head(x, y, fit, **kwargs)
    snapshots = {}
    b = fit_head(x, y, fit, snapshot_callback=lambda step, logits: snapshots.update({step:logits}), **kwargs)
    assert a == b
    assert set(snapshots) == {0,25,50}


def test_full_diagnostic_report(tmp_path):
    torch.set_num_threads(1)
    g = torch.Generator().manual_seed(9)
    x = torch.randn(256, 4, generator=g)
    y = torch.arange(256) % 2 == 0
    fit = torch.arange(256) < 128
    panel = dict(schema=1, capture_step=50, world_size=16,
        target=y, fit_mask=fit, inner_mask=torch.arange(256) % 4 < 2,
        features=dict(fourier=x,raw_fourier=x), current_logits=torch.zeros(256))
    result = diagnostic.run_panel(panel, tmp_path, steps=300)
    assert len(result['arms']) == 4
    for arm in result['arms'].values():
        assert [r['step'] for r in arm['curve']] == [0,50,100,200,300]
        assert '50_to_200/holdout' in arm['transitions']
    saved = torch.load(tmp_path / 'sample_scores.pt', weights_only=True)
    assert saved['arms']['full_batch'][200].shape == (256,)
