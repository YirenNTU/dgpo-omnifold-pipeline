import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_h4_no_fourier_depth as launcher
from RL.DGPO_neutrino.omnifold_ztautau.stage import build_fit_config
from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import EvenetAdapterRatioClassifier, pack_event_inputs
from RL.DGPO_neutrino.omnifold_ztautau.test_ztautau_omnifold import _FakeZtautauBackbone, _event_batch_with_pair_context
from RL.DGPO_neutrino.omnifold_ztautau.ratio_fit import fit_density_ratio
from dataclasses import replace


@pytest.mark.parametrize('depth', [1, 2])
def test_resolved_no_fourier_model_and_fit(depth):
    _, cfg = launcher.validated_config(depth)
    fit = build_fit_config(cfg['dgpo']['adaptive_omnifold']['audit_fit'], n_train=20000, n_validation=4000)
    fit.validate()
    packed, spec = pack_event_inputs(_event_batch_with_pair_context(batch_size=32), include_pairwise_context=False)
    model = EvenetAdapterRatioClassifier(_FakeZtautauBackbone(), spec,
        decoder_hidden_dim=8, decoder_layers=depth, decoder_heads=2,
        periodic_pair_features=False, topology_fourier_embedding=False,
        train_backbone=True, adapter_bottleneck=4)
    assert len(model.bank.decoder.blocks) == depth
    assert model.bank.pairwise_feature_dim == 0
    assert not hasattr(model.bank, 'topology_encoder')
    assert not hasattr(model.bank, 'fusion')
    data = (packed, torch.randn(32, 4), torch.ones(32), packed, torch.randn(32, 4), torch.ones(32))
    rows = []
    fit_density_ratio(model, *data, replace(fit, steps=3, min_steps=0,
        batch_size=16, train_microbatch_size_per_rank=None, train_candidates_per_event=None, restore_best=False,
        representation_probe_rows=32, representation_probe_interval_steps=2,
        diagnostic_max_snapshots=0), 42, validation=data, progress_callback=rows.append)
    assert rows[0]['stability/representation_probe/initial/branches_complete'] == 1
    assert rows[0]['stability/representation_probe/initial/error'] == 0
    assert not any('fourier/' in k for k in rows[0])


def test_reject_accidental_fourier():
    _, cfg = launcher.validated_config(2)
    cfg['dgpo']['adaptive_omnifold']['audit_fit']['periodic_pair_features'] = True
    with patch.object(launcher, 'read_overlay_yaml', return_value=cfg):
        with pytest.raises(ValueError, match='no-Fourier'):
            launcher.validated_config(2)
