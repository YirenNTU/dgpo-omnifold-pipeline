import copy
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from .ratio_fit import RatioFitConfig, fit_density_ratio
from .stability_diagnostic import StabilityDiagnostic, diagnostic_modules, replay_snapshot


class Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Linear(4, 4)
        self.drop = nn.Dropout(.2)

    def forward(self, x):
        return x + self.drop(self.up(x))


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bank = nn.Module()
        self.bank.decoder = nn.Module()
        self.bank.decoder.norm = nn.LayerNorm(4, elementwise_affine=False)
        self.bank.decoder.attn = nn.MultiheadAttention(4, 2, dropout=.2, batch_first=True)
        self.bank.output = nn.Linear(4, 1)
        self.backbone = nn.Module()
        self.backbone.PET = nn.Module()
        self.backbone.PET.adapter = Adapter()

    def forward(self, condition, sample):
        x = self.backbone.PET.adapter(sample).unsqueeze(1)
        x = self.bank.decoder.norm(x)
        x = self.bank.decoder.attn(x, x, x, need_weights=False)[0]
        return self.bank.output(x).flatten()


def population():
    generator = torch.Generator().manual_seed(45)
    return (torch.zeros(16, 2), torch.randn(16, 4, generator=generator), torch.ones(16),
            torch.zeros(16, 2), torch.randn(16, 4, generator=generator), torch.ones(16))


def config(**kwargs):
    values = dict(steps=3, batch_size=8, learning_rate=.01, validation_interval_steps=2,
                  validation_patience_evaluations=10, restore_best=False, progress_interval_steps=1,
                  diagnostic_enabled=True, diagnostic_interval_steps=1,
                  diagnostic_probe_rows=4, diagnostic_gradient_threshold=1e20)
    values.update(kwargs)
    return RatioFitConfig(**values)


def test_diagnostics_do_not_change_updates_rng_or_validation_schedule(tmp_path):
    torch.manual_seed(56)
    original = TinyClassifier()
    models, results, rows, rng = [], [], [], []
    for enabled in (False, True):
        model = copy.deepcopy(original)
        torch.manual_seed(78)
        history = []
        results.append(fit_density_ratio(model, *population(), config(diagnostic_enabled=enabled,
                                         diagnostic_snapshot_dir=str(tmp_path), diagnostic_gradient_threshold=1e-12),
                                         23, validation=population(), progress_callback=history.append))
        models.append(model)
        rows.append(history)
        rng.append(torch.get_rng_state())
    for name, value in models[0].state_dict().items():
        assert torch.equal(value, models[1].state_dict()[name]), name
    assert torch.equal(*rng)
    assert results[0].validation_history == results[1].validation_history
    assert [r['training_loss'] for r in rows[0]] == [r['training_loss'] for r in rows[1]]
    row = rows[1][-1]
    assert 'stability/probe/bce_delta' in row
    assert any('pre_norm_variance_min' in key for key in row)
    assert any('attention_entropy' in key for key in row)
    assert any('residual_input_rms_ratio' in key for key in row)
    assert any('input_gradient_rms' in key for key in row)
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in models[1].modules())


def test_probe_restores_unregistered_backbone_modes_buffers_rng():
    class Shared(TinyClassifier):
        def __init__(self):
            super().__init__()
            body = self.backbone
            del self.backbone
            object.__setattr__(self, 'backbone', body)
            self.backbone.register_buffer('counter', torch.tensor(0.))

        def train(self, mode=True):
            super().train(mode)
            self.backbone.train(mode)
            return self

        def forward(self, c, z):
            self.backbone.counter.add_(1)
            return super().forward(c, z)

    model = Shared().train()
    model.backbone.PET.adapter.drop.eval()
    modes = {n: m.training for n, m in diagnostic_modules(model).items()}
    state = torch.get_rng_state()
    diagnostic = StabilityDiagnostic(model, population(), config(), rank=0, world=1, seed=23)
    a, b = diagnostic.probe_scores(), diagnostic.probe_scores()
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert torch.equal(state, torch.get_rng_state())
    assert model.backbone.counter.item() == 0
    assert modes == {n: m.training for n, m in diagnostic_modules(model).items()}


def test_snapshot_replays_exact_dropout_update_and_is_bounded(tmp_path):
    torch.manual_seed(56)
    model = TinyClassifier()
    original = copy.deepcopy(model)
    rows = []
    fit_density_ratio(model, *population(), config(steps=1, diagnostic_gradient_threshold=1e-12,
                      diagnostic_snapshot_dir=str(tmp_path), diagnostic_max_snapshots=1),
                      23, validation=population(), progress_callback=rows.append)
    paths = list(tmp_path.glob('*/snapshot.pt'))
    assert len(paths) == 1
    payload = torch.load(paths[0], weights_only=False)
    assert payload['reason'] == 'gradient_spike'
    replay_snapshot(original, payload)
    for name, value in model.state_dict().items():
        assert torch.equal(value, original.state_dict()[name]), name
    assert rows[-1]['stability/snapshots_saved_local'] == 1


def test_post_update_snapshot_keeps_pre_update_parameters(tmp_path):
    model = TinyClassifier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    diagnostic = StabilityDiagnostic(model, population(), config(diagnostic_snapshot_dir=str(tmp_path)), rank=0, world=1, seed=1)
    diagnostic.begin(1, population(), optimizer, 16)
    diagnostic.close_hooks()
    original = copy.deepcopy(model)
    diagnostic.prepare_update()
    with torch.no_grad():
        model.bank.output.bias.add_(100.)
    diagnostic.after_update()
    payload = torch.load(next(tmp_path.glob('*/snapshot.pt')), weights_only=False)
    assert payload['reason'] == 'probe_bce_jump'
    assert torch.equal(payload['pre_update_parameters']['bank.output.bias'], original.bank.output.bias)


def test_requires_validation():
    with pytest.raises(ValueError, match='validation'):
        fit_density_ratio(TinyClassifier(), *population(), config(), 23)


def test_nonfinite_saves_before_failure_and_removes_hooks(tmp_path):
    model = TinyClassifier()
    model.bank.output.weight.register_hook(lambda grad: torch.full_like(grad, float('nan')))
    with pytest.raises(FloatingPointError):
        fit_density_ratio(model, *population(), config(diagnostic_snapshot_dir=str(tmp_path)),
                          23, validation=population())
    files = list(tmp_path.glob('*/snapshot.pt'))
    assert len(files) == 1
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
    payload = torch.load(files[0], weights_only=False)
    assert payload['step'] == 1
    assert all(torch.isfinite(v).all() for v in payload['pre_update_parameters'].values())


def test_snapshot_quota(tmp_path):
    fit_density_ratio(TinyClassifier(), *population(), config(steps=3, diagnostic_gradient_threshold=1e-12,
                      diagnostic_snapshot_dir=str(tmp_path), diagnostic_max_snapshots=1), 23, validation=population())
    assert len(list(tmp_path.glob('*/snapshot.pt'))) == 1


def test_evenet_shared_backbone_snapshot_replay(tmp_path):
    from .test_ztautau_omnifold import _FakeZtautauBackbone, _event_batch
    from .evenet_ratio import EvenetAdapterRatioClassifier, pack_event_inputs
    torch.manual_seed(80)
    packed, spec = pack_event_inputs(_event_batch(batch_size=8))
    def build():
        return EvenetAdapterRatioClassifier(_FakeZtautauBackbone(), spec,
            train_backbone=True, decoder_hidden_dim=8, decoder_layers=1,
            decoder_heads=2, adapter_bottleneck=4, head_dropout=.2)
    model = build()
    data = (packed, torch.randn(8, 4), torch.ones(8), packed, torch.randn(8, 4), torch.ones(8))
    fit_density_ratio(model, *data, config(steps=1, train_microbatch_size_per_rank=2,
        diagnostic_snapshot_dir=str(tmp_path), diagnostic_gradient_threshold=1e-12),
        23, validation=data)
    payload = torch.load(next(tmp_path.glob('*/snapshot.pt')), weights_only=False)
    replayed = build()
    replay_snapshot(replayed, payload)
    for name, value in model.state_dict().items():
        assert torch.equal(value, replayed.state_dict()[name]), name
    for name, value in model.backbone.state_dict().items():
        assert torch.equal(value, replayed.backbone.state_dict()[name]), name


def _distributed_worker(rank, rendezvous, directory):
    torch.set_num_threads(1)
    torch.distributed.init_process_group('gloo', init_method=f'file://{rendezvous}', rank=rank, world_size=2)
    try:
        torch.manual_seed(56)
        model = TinyClassifier()
        rows = []
        fit_density_ratio(model, *population(), config(steps=1, diagnostic_gradient_threshold=1e-12,
                          diagnostic_snapshot_dir=directory, diagnostic_max_snapshots=1),
                          23, validation=population(), progress_callback=rows.append)
        assert rows[-1]['stability/snapshots_saved_local'] == 1
        assert 'stability/probe/bce_before' in rows[-1]
        values = [None, None]
        torch.distributed.all_gather_object(values, rows[-1]['stability/probe/bce_after'])
        assert values[0] == values[1]
        payload = torch.load(next(Path(directory).glob(f'*rank{rank}-*/snapshot.pt')), weights_only=False)
        replayed = copy.deepcopy(model)
        replay_snapshot(replayed, payload)
        for name, value in model.state_dict().items():
            assert torch.equal(value, replayed.state_dict()[name]), name
        # Only rank 1 has a local spike; both ranks must save the same boundary.
        opt = torch.optim.AdamW(model.parameters())
        diagnostic = StabilityDiagnostic(model, population(), config(diagnostic_snapshot_dir=directory + '-rankspike',
                                          diagnostic_gradient_threshold=10.), rank=rank, world=2, seed=1)
        diagnostic.begin(1, population(), opt, 16)
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter) * (100. if rank else 0.)
        diagnostic.after_backward(torch.tensor(0.))
        assert diagnostic.saved == 1
        assert diagnostic.metrics['stability/gradient_spike'] == 1.
        assert (diagnostic.local_norm == 0.) == (rank == 0)
    finally:
        torch.distributed.destroy_process_group()


def test_two_rank_collectives_and_per_rank_snapshots(tmp_path):
    torch.multiprocessing.spawn(_distributed_worker,
        args=(str(tmp_path / 'rendezvous'), str(tmp_path / 'snapshots')), nprocs=2, join=True)
    files = list((tmp_path / 'snapshots').glob('*/snapshot.pt'))
    assert len(files) == 2
    assert {torch.load(p, weights_only=False)['rank'] for p in files} == {0, 1}
