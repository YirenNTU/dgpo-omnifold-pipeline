"""CPU fixtures only: ratio algebra, split isolation, best-val reload, 16-rank math."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from scripts.tau_residual_ratio import (
    BestValidation, compose, log_normalizer, split_events, make_data,
    epoch_batches, training_worker, load_stack,
)
from scripts.audit_conditional_tau_ratio import class_weights
from scripts.run_tau_residual_ratio import read_settings, fit_config, ROOT, merge_panel, physics_report
from scripts.train_conditional_spin_ratio import build_classifier, score_pair


def model_config():
    return dict(condition_dim=7, candidate_dim=29, hidden=16, dropout=0., relative_dim=6,
                head_kind='film', head_depth=3, condition_hidden=256, condition_width=256,
                packing_spec={}, relative_preprocessing={}, ratio_objective='bce', ratio_bound=None)


def test_oracle_residual_recovers_target_and_normalization_is_essential():
    p = np.array([.1, .5, .4]); q = np.array([.5, .3, .2]); w1 = np.array([.2, 1., 4.])
    z1 = np.dot(q, w1)
    q1 = q*w1/z1
    s2 = np.log(p/q1)
    actual = compose(np.log(w1), s2, np.log(z1))
    np.testing.assert_allclose(np.exp(actual), p/q)
    assert log_normalizer(q, np.log(w1)) == pytest.approx(np.log(z1))
    # Balanced BCE has zero expected score gradient at the residual oracle.
    s = torch.tensor(s2, requires_grad=True)
    loss = .5*(torch.tensor(p)*F.softplus(-s)+torch.tensor(q1)*F.softplus(s)).sum()
    loss.backward()
    torch.testing.assert_close(s.grad, torch.zeros_like(s), atol=1e-15, rtol=0)


def test_cap_is_on_product_not_residual_and_constant_is_frozen():
    base = np.log(np.array([20., .1]))
    residual = np.log(np.array([3., 100.]))
    np.testing.assert_allclose(np.exp(compose(base, residual, 0)), [30., 10.])
    np.testing.assert_allclose(compose(base, np.full(2, .7), .7), base)
    np.testing.assert_allclose(compose(np.tile(base, (3, 1)), np.tile(residual, (3, 1)), 0),
                               np.tile(np.log([30., 10.]), (3, 1)))
    with pytest.raises(ValueError):
        compose(base, residual[:1], 0)
    with pytest.raises(ValueError):
        compose(base, residual, float('nan'))


def test_fixed_class_normalization_does_not_normalize_each_event_or_batch():
    base = np.array([1., 2., 3., 4.]); s = np.log([.1, 1., 10., 30.])
    p, q = class_weights(base, s, True)
    assert p.mean() == pytest.approx(1) and q.mean() == pytest.approx(1)
    np.testing.assert_allclose(q/p, np.exp(s-log_normalizer(base, s)))
    assert q[:2].mean() != pytest.approx(1)


def fixture_arrays(n=200):
    rng = np.random.default_rng(8)
    return dict(source_ids=np.arange(n).astype(str), condition=rng.normal(size=(n, 7)).astype('float32'),
        candidate_truth=rng.normal(size=(n, 29)).astype('float32'),
        candidate_generated=rng.normal(size=(n, 29)).astype('float32'),
        event_weight=np.ones(n), split=np.r_[np.zeros(n//3), np.ones(n//3), np.full(n-2*(n//3), 2)])


def test_event_splits_exclude_base_fit_and_residual_test_leakage():
    a = fixture_arrays(900)
    residual = make_data(a, np.zeros(900), 71)
    audit = make_data(a, np.zeros(900), 72, audit=True)
    assert set(residual['source_ids']) == set(a['source_ids'][a['split'] == 1])
    assert set(audit['source_ids']) == set(a['source_ids'][a['split'] == 2])
    assert set(audit['source_ids']).isdisjoint(residual['source_ids'])
    ids = a['source_ids']
    np.testing.assert_array_equal(split_events(ids[::-1], 71)[::-1], split_events(ids, 71))
    with pytest.raises(ValueError, match='unique'):
        split_events(np.array(['x', 'x']), 71)
    # Neither event IDs nor Cij enters the classifier inputs.
    assert set(residual) == {'source_ids', 'condition', 'candidate_truth', 'candidate_generated', 'event_weight', 'log_ratio', 'split'}


def test_best_val_is_not_patience_best_or_last_or_minimum_step_checkpoint():
    s = BestValidation(patience=2, min_delta=.01, min_steps=100)
    assert s.update(.5, 1) == (True, False)
    assert s.update(.4999, 100) == (True, False)
    assert s.update(.4998, 101) == (True, True)  # tiny true improvement STILL saved
    assert s.best == .4998 and s.patience_best == .5
    with pytest.raises(ValueError, match='Nonfinite'):
        s.update(float('nan'), 102)


@pytest.mark.parametrize('n,batch', [(7, 1024), (35, 2), (64, 2)])
def test_16_rank_padding_has_exactly_once_event_mass_and_global_loss(n, batch):
    all_batches = [list(epoch_batches(n, 16, batch, rank, 42, 0)) for rank in range(16)]
    assert len(set(map(len, all_batches))) == 1
    visits = np.zeros(n, int)
    values = np.linspace(.1, 2., n)
    for j in range(len(all_batches[0])):
        losses = []
        global_idx = []
        for rows in all_batches:
            idx, valid, total = rows[j]
            np.add.at(visits, idx[valid], 1)
            losses.append(np.sum(values[idx]*valid)*16/total)
            global_idx.extend(idx[valid])
        assert np.mean(losses) == pytest.approx(np.mean(values[global_idx]))
    np.testing.assert_array_equal(visits, 1)


def test_config_preserves_architecture_and_best_validation_selection(tmp_path):
    s = read_settings(ROOT/'config/conditional_tau_residual_ratio_10pct.yaml')
    a = fixture_arrays()
    cfg = fit_config(s, model_config(), a, tmp_path, 'residual', tmp_path/'inputs.npz', .2)
    assert cfg['workers'] == 16 and cfg['batch_size'] == 1024
    assert cfg['ratio_bound'] is None and cfg['cumulative_cap'] == 30
    assert cfg['condition_width'] == 256 and cfg['head_depth'] == 3
    assert cfg['selector'] == 'minimum_validation_weighted_bce'
    assert cfg['grad_clip'] is None and cfg['min_steps'] == 1000
    assert cfg['policy_updates'] == cfg['backbone_updates'] == cfg['baseline_refits'] == 0


def test_actual_training_worker_restores_best_before_test(tmp_path):
    import ray.train.torch
    a = fixture_arrays(60)
    a['log_ratio'] = np.linspace(-1, 1, 60)
    np.savez(tmp_path/'data.npz', **a)
    cfg = dict(model_config(), prepared=str(tmp_path/'data.npz'), checkpoint=str(tmp_path/'best.pt'),
        workers=1, batch_size=8, seed=42, lr=2e-4, min_lr=1e-5, weight_decay=.001,
        epochs=3, patience=3, min_delta=.1, min_steps=0, grad_clip=None)
    from scripts.audit_conditional_tau_ratio import metrics as real_metrics
    calls, reports = [], []
    def metric(*args):
        value = real_metrics(*args)
        value['bce'] = [.5, .4999, .6][len(calls)]
        calls.append(value)
        return value
    def gather(value, out, dst=0):
        out[0] = value
    with patch('ray.train.get_context', return_value=SimpleNamespace(get_world_rank=lambda: 0, get_world_size=lambda: 1)), \
         patch('ray.train.torch.get_device', return_value=torch.device('cpu')), \
         patch('ray.train.torch.prepare_model', side_effect=lambda m: m), \
         patch('ray.train.report', side_effect=reports.append), \
         patch('torch.distributed.all_reduce'), patch('torch.distributed.broadcast'), \
         patch('torch.distributed.barrier'), patch('torch.distributed.gather_object', side_effect=gather), \
         patch('scripts.tau_residual_ratio.metrics', side_effect=metric):
        training_worker(cfg)
    best = torch.load(cfg['checkpoint'], weights_only=True)
    assert best['epoch'] == 2 and best['val_bce'] == .4999
    assert len(reports) == 3 and reports[-1]['best_val_bce'] == .4999
    model = build_classifier(best); model.load_state_dict(best['state_dict'])
    test = a['split'] == 2
    p, q = score_pair(model, a['condition'][test], a['candidate_truth'][test], a['candidate_generated'][test], 'cpu', 8)
    with np.load(tmp_path/'test-rank-00.npz') as f:
        np.testing.assert_allclose(f['truth_logits'], p)
        np.testing.assert_allclose(f['generated_logits'], q)
    status = json.loads((tmp_path/'fit_status.json').read_text())
    assert status['epochs'] == 3 and status['best_epoch'] == 2


def test_saved_stack_replays_product_and_rejects_changed_constant(tmp_path):
    cfg = model_config()
    torch.manual_seed(42)
    base = build_classifier(dict(cfg, ratio_bound=30)).eval()
    residual = build_classifier(cfg).eval()
    bp, rp = tmp_path/'base.pt', tmp_path/'residual.pt'
    torch.save(dict(cfg, ratio_bound=30, state_dict=base.state_dict()), bp)
    torch.save(dict(cfg, fit_log_z=.3, cumulative_cap=30, state_dict=residual.state_dict()), rp)
    stack = dict(base_checkpoint=str(bp), residual_checkpoint=str(rp), fit_log_z=.3, cap=30)
    (tmp_path/'stack.json').write_text(json.dumps(stack))
    model = load_stack(tmp_path/'stack.json')
    c, t = torch.randn(4, 7), torch.randn(4, 29)
    expected = compose(base(c, t).detach().numpy(), residual(c, t).detach().numpy(), .3)
    np.testing.assert_allclose(model(c, t).detach().numpy(), expected, atol=1e-12)
    stack['fit_log_z'] = 0
    (tmp_path/'stack.json').write_text(json.dumps(stack))
    with pytest.raises(ValueError, match='constants'):
        load_stack(tmp_path/'stack.json')


def test_missing_or_duplicate_panel_shards_rejected(tmp_path):
    ids = np.array(['a', 'b', 'c', 'd'])
    for rank, idx in enumerate(([0, 2], [1, 3])):
        np.savez(tmp_path/f'panel-{rank:02d}.npz', positions=idx, source_ids=ids[idx],
                 logits=np.zeros((2, 64)), residual_logits=np.ones((2, 64)), max_base_replay_error=0.)
    q, r, p = merge_panel(tmp_path, ids, 2)
    assert q.shape == (4, 64) and (r == 1).all() and p == 0
    np.savez(tmp_path/'panel-01.npz', positions=[1, 1], source_ids=ids[[1, 1]],
             logits=np.zeros((2, 64)), residual_logits=np.ones((2, 64)), max_base_replay_error=0.)
    with pytest.raises(ValueError, match='Missing, repeated'):
        merge_panel(tmp_path, ids, 2)


def test_physics_comparison_identity_and_no_truth_checkpoint_selection(tmp_path):
    from scripts.run_tau_bounded_ratio import bounded_panel_report
    rng = np.random.default_rng(3); n = 24
    panel, base = tmp_path/'panel', tmp_path/'base'
    panel.mkdir(); base.mkdir()
    inputs = dict(source_ids=np.arange(n).astype(str), weight=np.ones(n), truth_cij=rng.normal(size=(n, 9)),
                  category=np.full(n, 11), visible_pt_sum=np.arange(n))
    data = dict(logits=rng.normal(size=(n, 64)), cij=rng.normal(size=(n, 64, 9)))
    old = np.minimum(data['logits'], np.log(30))
    np.savez(panel/'inputs.npz', **inputs); np.savez(panel/'samples_and_scores.npz', **data)
    np.savez(base/'fixed_panel_scores.npz', source_ids=inputs['source_ids'], logits=old)
    np.savez(base/'test_scores.npz', source_ids=inputs['source_ids'], log_ratio=old[:, 0], generated_logits=old[:, 0])
    prior = bounded_panel_report(inputs, data, old, 20, 42)
    (base/'fixed_K64_report.json').write_text(json.dumps(prior))
    s = read_settings(ROOT/'config/conditional_tau_residual_ratio_10pct.yaml')
    s.update(panel_directory=str(panel), wide_directory=str(base), bootstrap=20)
    report = physics_report(s, dict(condition_pt_edges=[8, 16]), tmp_path, old)
    assert report['source_endpoints_verified'] and not report['qualified_improvement']
    assert set(report['arms']) == {'unweighted', 'baseline', 'residual'}
    for key in ('offdiagonal', 'diagonal', 'total'):
        assert report['comparisons'][key]['change'] == 0
        np.testing.assert_allclose(report['comparisons'][key]['ci95'], 0)


def test_prepare_cli_does_not_launch_training(monkeypatch):
    from scripts import run_tau_residual_ratio as launcher
    monkeypatch.setattr(launcher, 'prepare', lambda _: (None, None))
    def fail(*args):
        raise AssertionError('prepare must not launch')
    monkeypatch.setattr(launcher, 'execute', fail)
    monkeypatch.setattr('sys.argv', ['run', str(ROOT/'config/conditional_tau_residual_ratio_10pct.yaml'), 'prepare'])
    launcher.main()


def _distributed_training(rank, rendezvous, directory):
    """Actual gloo/DDP/gather checkpoint test, not a simulation of collectives."""
    import ray.train.torch
    from torch import distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    dist.init_process_group('gloo', init_method='file://'+rendezvous, rank=rank, world_size=2)
    directory = Path(directory)
    try:
        cfg = dict(model_config(), prepared=str(directory/'data.npz'), checkpoint=str(directory/'best.pt'),
            workers=2, batch_size=7, seed=42, lr=2e-4, min_lr=1e-5, weight_decay=.001,
            epochs=2, patience=3, min_delta=.001, min_steps=0, grad_clip=None)
        reports = []
        with patch('ray.train.get_context', return_value=SimpleNamespace(get_world_rank=lambda: rank, get_world_size=lambda: 2)), \
             patch('ray.train.torch.get_device', return_value=torch.device('cpu')), \
             patch('ray.train.torch.prepare_model', side_effect=DistributedDataParallel), \
             patch('ray.train.report', side_effect=reports.append):
            training_worker(cfg)
        (directory/f'reports-{rank}.json').write_text(json.dumps(reports))
    finally:
        dist.destroy_process_group()


def test_real_two_rank_training_validation_and_best_inference(tmp_path):
    import torch.multiprocessing as mp
    from scripts.audit_conditional_tau_ratio import merge_scores as merge_audit
    a = fixture_arrays(63)
    # 21 training conditions: second global batch has unequal real counts (4/3).
    a['log_ratio'] = np.linspace(-1, 1, 63)
    np.savez(tmp_path/'data.npz', **a)
    mp.spawn(_distributed_training, args=(str(tmp_path/'rendezvous'), str(tmp_path)), nprocs=2, join=True)
    reports = [json.loads((tmp_path/f'reports-{rank}.json').read_text()) for rank in range(2)]
    assert reports[0] == reports[1]
    assert reports[0][-1]['optimizer_steps'] == 4
    best = torch.load(tmp_path/'best.pt', weights_only=True)
    model = build_classifier(best); model.load_state_dict(best['state_dict'])
    ti = a['split'] == 2
    p, q = score_pair(model, a['condition'][ti], a['candidate_truth'][ti], a['candidate_generated'][ti], 'cpu', 7)
    actual = merge_audit(tmp_path, a['source_ids'][ti], 2)
    np.testing.assert_allclose(actual[0], p, atol=1e-6)
    np.testing.assert_allclose(actual[1], q, atol=1e-6)
