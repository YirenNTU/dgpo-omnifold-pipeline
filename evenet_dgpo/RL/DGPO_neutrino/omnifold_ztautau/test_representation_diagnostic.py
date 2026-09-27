import copy
from dataclasses import replace

import pytest
import torch
from torch import nn

from .evenet_ratio import AdaLNZeroCandidateDecoder
from .ratio_fit import RatioFitConfig, fit_density_ratio
from .representation_diagnostic import binary_auc, context_partition, ridge_readout


class BranchClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bank = nn.Module()
        self.bank.decoder = AdaLNZeroCandidateDecoder(token_dim=4, event_dim=4, hidden_dim=4,
                                                      num_layers=1, num_heads=2, dropout=.2)
        self.bank.topology_encoder = nn.Sequential(nn.LayerNorm(4), nn.Linear(4, 4), nn.GELU(), nn.Dropout(.2))
        self.bank.fusion = nn.Sequential(nn.LayerNorm(12), nn.Linear(12, 4), nn.GELU(), nn.Dropout(.2))
        self.bank.output = nn.Linear(4, 1)
        nn.init.zeros_(self.bank.output.weight)
        nn.init.zeros_(self.bank.output.bias)

    def forward(self, c, z):
        decoder = self.bank.decoder(candidate_tokens=torch.stack((z, -z), 1),
            event_token=c, memory_tokens=torch.stack((c, c), 1),
            memory_mask=torch.ones(len(c), 2, 1, dtype=torch.bool, device=c.device))
        fourier = self.bank.topology_encoder(z)
        return self.bank.output(self.bank.fusion(torch.cat((decoder.flatten(1), fourier), 1))).flatten()


def population(n=96):
    g = torch.Generator().manual_seed(16)
    c = torch.randn(n, 4, generator=g)
    return c, torch.randn(n, 4, generator=g) + .5, torch.ones(n), c, torch.randn(n, 4, generator=g) - .5, torch.ones(n)


def config():
    return RatioFitConfig(steps=3, batch_size=16, learning_rate=.002, restore_best=False,
        validation_interval_steps=2, validation_patience_evaluations=20, progress_interval_steps=1,
        diagnostic_enabled=True, diagnostic_interval_steps=10, diagnostic_probe_rows=4,
        diagnostic_max_snapshots=0, representation_diagnostic_enabled=True,
        representation_probe_rows=96, representation_probe_interval_steps=2)


def test_ridge_separability_null_ties_and_context_split():
    g = torch.Generator().manual_seed(92)
    c = torch.randn(256, 4, generator=g)
    split = torch.cat((context_partition(c), context_partition(c)))
    assert torch.equal(split[:256], split[256:])
    y = torch.arange(512) < 256
    x = torch.randn(512, 4, generator=g)
    x[:, 0] += y.double() * 6 - 3
    result = ridge_readout(x, y, split)
    assert result['valid'] == 1 and result['holdout_auc'] > .98
    assert binary_auc(torch.zeros(512), y) == .5
    null = ridge_readout(torch.randn(512, 4, generator=g), y, split)
    assert .3 < null['holdout_auc'] < .7
    assert ridge_readout(x, y, torch.ones(512, dtype=torch.bool))['valid'] == 0


def test_probes_do_not_change_training_rng_or_buffers():
    torch.manual_seed(7)
    original = BranchClassifier()
    models, results, histories, states = [], [], [], []
    for enabled in (False, True):
        model = copy.deepcopy(original)
        torch.manual_seed(23)
        rows = []
        result = fit_density_ratio(model, *population(),
            replace(config(), representation_diagnostic_enabled=enabled, representation_path_enabled=enabled), 42,
            validation=population(), progress_callback=rows.append)
        models.append(model); results.append(result); histories.append(rows)
        states.append(torch.get_rng_state())
    assert torch.equal(*states)
    for k, v in models[0].state_dict().items():
        assert torch.equal(v, models[1].state_dict()[k]), k
    assert results[0].validation_history == results[1].validation_history
    assert [r['training_loss'] for r in histories[0]] == [r['training_loss'] for r in histories[1]]
    row = histories[1][0]
    assert row['stability/representation_probe/initial/error'] == 0
    for branch in ('raw_fourier', 'normalized_fourier', 'fourier', 'decoder', 'concat', 'fusion'):
        assert row[f'stability/representation_probe/initial/{branch}/valid'] == 1
        assert row[f'stability/representation_probe/initial/{branch}/cv/valid'] == 1
    assert row['stability/representation_probe/initial/current_head/panel_auc'] == .5
    assert row['stability/layer/representation/fourier/gradient_rms/rankmean'] == 0
    assert row['stability/layer/representation/bank.decoder.blocks.0.modulation/gate_cross_rms/rankmean'] == 0
    assert 'stability/representation/parameter/bank.output.weight/rms' in row
    assert histories[1][1]['stability/representation_probe/step'] == 2
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in models[1].modules())


def test_invalid_config():
    with pytest.raises(ValueError, match='diagnostic_enabled'):
        replace(config(), diagnostic_enabled=False).validate()
    with pytest.raises(ValueError, match='positive integers'):
        replace(config(), representation_probe_rows=0).validate()


def test_real_evenet_bank_with_shared_backbone_is_observational():
    from .test_ztautau_omnifold import _FakeZtautauBackbone, _event_batch_with_pair_context
    from .evenet_ratio import EvenetAdapterRatioClassifier, pack_event_inputs
    torch.manual_seed(19)
    packed, spec = pack_event_inputs(_event_batch_with_pair_context(batch_size=32), include_pairwise_context=True)
    data = (packed, torch.randn(32, 4), torch.ones(32), packed, torch.randn(32, 4), torch.ones(32))
    models, rngs = [], []
    for enabled in (False, True):
        torch.manual_seed(38)
        model = EvenetAdapterRatioClassifier(_FakeZtautauBackbone(), spec,
            train_backbone=True, decoder_hidden_dim=8, decoder_layers=1, decoder_heads=2,
            adapter_bottleneck=4, head_dropout=.2, periodic_pair_features=True,
            topology_fourier_embedding=True, topology_hidden_dim=8, topology_embedding_dim=4,
            topology_fusion_hidden_dim=8)
        rows = []
        fit_density_ratio(model, *data, replace(config(), steps=2, representation_diagnostic_enabled=enabled, representation_path_enabled=enabled),
                          42, validation=data, progress_callback=rows.append)
        if enabled:
            assert rows[-1]['stability/representation_probe/error'] == 0
            assert rows[-1]['stability/representation_probe/branches_complete'] == 1
            assert rows[-1]['stability/representation_probe/path_complete'] == 1
        models.append(model); rngs.append(torch.get_rng_state())
    assert torch.equal(*rngs)
    for bank in ('', 'backbone'):
        a, b = (getattr(m, bank) if bank else m for m in models)
        for name, value in a.state_dict().items():
            assert torch.equal(value, b.state_dict()[name]), (bank, name)


def _worker(rank, rendezvous):
    torch.set_num_threads(1)
    torch.distributed.init_process_group('gloo', init_method=f'file://{rendezvous}', rank=rank, world_size=2)
    try:
        histories, models, states = [], [], []
        for enabled in (False, True):
            torch.manual_seed(7)
            model = BranchClassifier()
            rows = []
            fit_density_ratio(model, *population(), replace(config(), steps=2, representation_diagnostic_enabled=enabled, representation_path_enabled=enabled,
                              representation_export_dir=rendezvous + '_panels' if enabled else None),
                              42, validation=population(), progress_callback=rows.append)
            histories.append(rows); models.append(model); states.append(torch.get_rng_state())
        assert torch.equal(*states)
        for name, value in models[0].state_dict().items():
            assert torch.equal(value, models[1].state_dict()[name]), name
        assert histories[1][-1]['stability/representation_probe/error'] == 0
        value = histories[1][-1]['stability/representation_probe/raw_fourier/cv/holdout_auc']
        gathered = [None, None]
        torch.distributed.all_gather_object(gathered, value)
        assert gathered[0] == gathered[1]
        panel = torch.load(rendezvous + '_panels/panel_step0000.pt', weights_only=True)
        assert panel['world_size'] == 2
        assert panel['capture_step'] == 0
        assert len(panel['target']) == 192
        assert {'raw_fourier', 'fourier', 'fusion'} <= panel['features'].keys()
    finally:
        torch.distributed.destroy_process_group()


def test_two_rank_probe_and_training_equivalence(tmp_path):
    torch.multiprocessing.spawn(_worker, args=(str(tmp_path / 'rendezvous'),), nprocs=2, join=True)


def test_nested_ridge_never_selects_on_holdout():
    from .representation_diagnostic import nested_ridge_readout
    g = torch.Generator().manual_seed(3)
    c = torch.randn(256, 4, generator=g)
    fit = torch.cat([context_partition(c)] * 2)
    inner = torch.cat([context_partition(c, byte=1)] * 2)
    y = torch.arange(512) < 256
    x = torch.randn(512, 8, generator=g)
    x[:, 0] += y.float() * 4
    a = nested_ridge_readout(x, y, fit, inner)
    x[~fit] *= -100
    y[~fit] = ~y[~fit]
    b = nested_ridge_readout(x, y, fit, inner)
    assert a['valid'] == b['valid'] == 1
    assert a['selected_lambda'] == b['selected_lambda']
    for key in a:
        if key.endswith('/inner_auc'):
            assert a[key] == b[key]
    assert nested_ridge_readout(x, y, fit, torch.ones_like(inner))['valid'] == 0
