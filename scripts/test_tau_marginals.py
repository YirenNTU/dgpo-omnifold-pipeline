import json
import numpy as np
from RL.DGPO_neutrino.diagnostics.tau_marginals import monitor, observables, probability, distances


def test_tail_and_distance():
    p = probability(np.array([-2., .5, 2.]), np.array([1., 2., 1.]), np.array([0., 1.]))
    np.testing.assert_allclose(p, [.25, .5, .25])
    assert distances(p, p) == {'tv': 0., 'jsd': 0.}
    assert distances(np.array([1., 0.]), np.array([0., 1.]))['tv'] == 1.


def test_monitor_fixed_baseline(tmp_path):
    n = 100
    rng = np.random.default_rng(42)
    truth = rng.normal(0, .03, (n, 2, 2))
    panel = dict(source_ids=np.arange(n), truth_deltas=truth, event_weight=np.ones(n),
                 visible_a=np.tile([4., 1., 1., 1.], (n, 1)),
                 visible_b=np.tile([4., -1., -1., -1.], (n, 1)))
    draws = np.stack([truth, truth+10], axis=1)
    source = tmp_path/'baseline.npz'
    np.savez(source, source_ids=panel['source_ids'], deltas=draws)
    logs, paths = monitor(panel, draws, source, tmp_path)
    assert len(paths) == 3 and all(p.is_file() for p in paths.values())
    assert all(v == 0 for k, v in logs.items() if '/tv/current/' in k)
    before = json.loads((tmp_path/'marginals.json').read_text())['histograms']
    changed = draws.copy(); changed[:, 0] += .2
    monitor(panel, changed, source, tmp_path)
    after = json.loads((tmp_path/'marginals.json').read_text())['histograms']
    assert all(before[k]['edges'] == after[k]['edges'] for k in before)
    wrapped = truth.copy(); wrapped[..., 1] += 2*np.pi
    np.testing.assert_allclose(observables(panel, wrapped)['target/tau_a_delta_phi'], truth[:, 0, 1], atol=1e-14)
    logs, paths = monitor(panel, draws, tmp_path/'missing.npz', tmp_path)
    assert logs['tau/marginal/baseline_available'] == 0 and not paths
