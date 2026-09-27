from dataclasses import replace

import pytest
import torch

from experiments.dgpo_toy import conditioning_placement as toy


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def small():
    return toy.Config(training=replace(toy.core.Config(), hidden=8, train_events=32,
        validation_events=32, test_events=64, train_probe_events=16, batch_size=16,
        patience=2, min_delta=100., max_epochs=2))


def panel(cfg):
    data = toy.core.make_dataset(cfg.training, cfg.case)
    p = toy.core.make_panel(data['validation'], 72, cfg.case, cfg.training)
    return data, p


def test_fork_exact_function_weights_and_adapter_initialization():
    cfg = small()
    source = toy.initialize(cfg, 17)
    models = toy.fork_models(source)
    _, p = panel(cfg)
    args = p['x'], p['t'], p['condition']
    expected = source(*args)
    for arm, model in models.items():
        assert torch.equal(expected, model(*args))
        for name, value in source.state_dict().items():
            assert torch.equal(value, model.state_dict()[name])
        assert model.adapter[0].weight.count_nonzero() > 0
        assert model.adapter[-1].weight.count_nonzero() == 0
        assert model.adapter[-1].weight.requires_grad == (arm != 'none')
    assert sum(p.numel() for p in models['early'].parameters() if p.requires_grad) == sum(
        p.numel() for p in models['late'].parameters() if p.requires_grad)


def test_late_does_not_change_body_forward_but_early_can():
    cfg = small()
    source = toy.initialize(cfg, 17)
    models = toy.fork_models(source)
    _, p = panel(cfg)
    args = p['x'], p['t'], p['condition']
    with torch.no_grad():
        models['early'].adapter[-1].weight.normal_(std=.1)
        models['late'].adapter.load_state_dict(models['early'].adapter.state_dict())
    body = source.representations(*args)[0]
    assert torch.equal(body, models['late'].representations(*args)[0])
    assert not torch.equal(body, models['early'].representations(*args)[0])
    assert not torch.equal(source(*args), models['late'](*args))


def test_paired_noise_no_truth_calls_and_all_backbones_train(monkeypatch):
    cfg = small()
    source = toy.initialize(cfg, 17)
    models = toy.fork_models(source)
    data, _ = panel(cfg)
    before = copy_state(source)
    seen = {a: [] for a in toy.ARMS}
    for arm, model in models.items():
        model.register_forward_pre_hook(lambda m, args, arm=arm: seen[arm].append(
            tuple(x.detach().clone() for x in args)))
    monkeypatch.setattr(toy.core, 'mean_function', lambda *a: pytest.fail('Truth leakage'))
    monkeypatch.setattr(toy.core, 'oracle_velocity', lambda *a: pytest.fail('Teacher loss'))
    opts = {a: toy.make_optimizer(m, cfg.training) for a, m in models.items()}
    assert all(not opt.state for opt in opts.values())
    toy.core.train_epoch(models, opts, data['train'], cfg.training, toy.core.generator(92))
    for arm, model in models.items():
        assert len(seen[arm]) == 2
        for a, b in zip(seen['none'], seen[arm]):
            assert all(torch.equal(x, y) for x, y in zip(a, b))
        assert any(p.grad is not None and p.grad.norm() > 0 for p in model.backbone.parameters())
        if arm != 'none':
            assert model.adapter[-1].weight.count_nonzero() > 0
            assert model.adapter[0].weight.grad.norm() > 0
    assert all(torch.equal(value, before[name]) for name, value in source.state_dict().items())


def copy_state(model):
    return {k: v.clone() for k, v in model.state_dict().items()}


def test_readiness_rejects_unlearned_and_too_good_sources():
    cfg = small()
    assert all(toy.readiness(1., {'velocity_mse': .39, 'oracle_excess_mse': .03}, cfg).values())
    assert not all(toy.readiness(1., {'velocity_mse': .9, 'oracle_excess_mse': .5}, cfg).values())
    assert not all(toy.readiness(1., {'velocity_mse': .35, 'oracle_excess_mse': .001}, cfg).values())


def test_continuation_selection_shared_budget_and_late_test(tmp_path, monkeypatch):
    cfg = small()
    source = toy.initialize(cfg, 17)
    data, _ = panel(cfg)
    rows, calls = [], []
    original = toy.core.make_panel
    def record(split, seed, case, config):
        calls.append((seed, len(rows)))
        return original(split, seed, case, config)
    monkeypatch.setattr(toy.core, 'make_panel', record)
    result = toy.continue_matched(tmp_path, data, cfg, 17, source, rows.append)
    assert result['initial_velocity_exact']
    assert result['epochs'] == 2
    assert len(rows) == 9
    assert calls[-1] == (74017, 9)
    for arm in toy.ARMS:
        best = min((r for r in rows if r['arm'] == arm), key=lambda r: r['validation']['velocity_mse'])
        assert result['endpoint'][arm]['selected_epoch'] == best['epoch']
        assert (tmp_path/f'best_{arm}.pt').exists()
        assert rows[toy.ARMS.index(arm)]['representation']['body_relative_rms_change'] == 0
    delta = result['endpoint']['late']['test']['velocity_mse']-result['endpoint']['early']['test']['velocity_mse']
    assert result['comparisons']['late_minus_early']['candidate_minus_baseline_mse'] == pytest.approx(delta)


def test_unsuitable_pretrain_does_not_launch_continuation(tmp_path, monkeypatch):
    cfg = small()
    monkeypatch.setattr(toy, 'continue_matched', lambda *a: pytest.fail('Invalid source continued'))
    report = toy.run(tmp_path, cfg)
    assert report['state'] == 'inconclusive_source_not_ready'
    assert 'continuation' not in report['results']['17']
    assert (tmp_path/'17/source.pt').exists()


def test_pretrain_ready_uses_selected_source_without_test(tmp_path, monkeypatch):
    cfg = small()
    data, _ = panel(cfg)
    original = toy.core.evaluate
    evaluations = []
    def controlled(model, panel):
        metrics, errors = original(model, panel)
        evaluations.append(len(panel['t']))
        metrics.update(velocity_mse=1. if len(evaluations) == 1 else .39,
                       oracle_excess_mse=.7 if len(evaluations) == 1 else .03)
        return metrics, errors
    monkeypatch.setattr(toy.core, 'evaluate', controlled)
    model, result = toy.pretrain(tmp_path, data, cfg, 17, lambda row: None)
    assert result['status'] == 'ready'
    assert result['selected_epoch'] == result['trained_epochs'] == 1
    assert evaluations == [32, 32]
    checkpoint = torch.load(tmp_path/'source.pt', weights_only=True)
    assert all(torch.equal(v, checkpoint['model'][k]) for k, v in model.state_dict().items())
