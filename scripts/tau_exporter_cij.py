"""Explicit root-exporter reconstruction for pre-unfolding Cij and Spin inputs.

This is a separate protocol. Do not substitute it into an already fitted
general-spin-tau-epoch287-v1 head or change the meaning of its saved metrics.
The upstream source root must be selected before importing the exporter.
"""
import importlib
from pathlib import Path
import sys

import awkward as ak
import numpy as np

from scripts.tau_tihsu_cij import AXES, channel_matrices, analyzing_powers

PROTOCOL = 'root-angle-exporter-target-p4-v2'


def reference_exporter(tree_ana_root):
    root = Path(tree_ana_root).resolve()
    repo = Path(__file__).resolve().parents[1]
    if not (root/'quantum/observables_builder.py').is_file():
        raise ValueError('Need the actual upstream observable builder')
    sys.path[:0] = [str(root), str(repo.parent), str(repo)]
    ob = importlib.import_module('quantum.observables_builder')
    if Path(ob.__file__).resolve() != root/'quantum/observables_builder.py':
        raise ValueError('Different quantum module already imported; start a fresh process')
    exporter = importlib.import_module('export_evenet_qi_inputs')
    if Path(exporter.__file__).resolve() != repo/'export_evenet_qi_inputs.py':
        raise ValueError('Unexpected exporter module')
    if exporter.build_observables is not ob.build_observables:
        raise ValueError('Exporter observable implementation differs')
    return exporter, ob


def event_records(visible_a, visible_b, *, deltas=None, truth_tau_a=None, truth_tau_b=None):
    """One candidate/event; no event filtering or inferred truth p4."""
    fields = {}
    n = len(visible_a)
    for name, values in [('lead_a_visible', visible_a), ('lead_b_visible', visible_b),
                         ('truth_tau_a', truth_tau_a), ('truth_tau_b', truth_tau_b)]:
        if values is None:
            continue
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (n, 4) or not np.isfinite(values).all():
            raise ValueError(f'{name} requires finite [N,4] p4 in E,px,py,pz order')
        fields[name+'_p4'] = ak.zip({k: values[:, j] for j, k in enumerate(('E', 'px', 'py', 'pz'))},
                                   with_name='Momentum4D')
    if deltas is not None:
        deltas = np.asarray(deltas, dtype=np.float64)
        if deltas.shape != (n, 2, 2) or not np.isfinite(deltas).all():
            raise ValueError('Need finite angular offsets [N,2,2]')
        for j, leg in enumerate(('a', 'b')):
            for i, coord in enumerate(('theta', 'phi')):
                fields[f'evenet_invisible_{leg}_{coord}'] = deltas[:, j, i]
            fields[f'evenet_invisible_{leg}_valid'] = np.ones(n, dtype=bool)
    return ak.Array(fields)


def mapped_observables(exporter, visible_a, visible_b, *, method, deltas=None,
                       truth_tau_a=None, truth_tau_b=None):
    if method not in ('target', 'evenet'):
        raise ValueError('Declare exporter target or evenet; no implicit reference')
    if method == 'target' and (truth_tau_a is None or truth_tau_b is None):
        raise ValueError('Exporter target needs complete truth tau p4, not truth offsets')
    if method == 'evenet' and deltas is None:
        raise ValueError('Exporter evenet needs candidate offsets')
    if method == 'evenet' and (truth_tau_a is not None or truth_tau_b is not None):
        raise ValueError('Do not pass truth p4 into generated-candidate mapping')
    events = event_records(visible_a, visible_b, deltas=deltas,
                          truth_tau_a=truth_tau_a, truth_tau_b=truth_tau_b)
    obs, valid, _ = exporter.method_observables(events, method)
    if not np.asarray(valid).all():
        raise ValueError('Invalid exporter events: supply the PI selection explicitly; no silent dropping')
    return {k: np.asarray(v, dtype=np.float64) for k, v in obs.items()}


def spin_inputs(obs):
    """The SAME mapped observables feed Spin15 and the nine Cij histograms."""
    names = [f'cos_theta_{leg}_{axis}' for leg in ('A', 'B') for axis in AXES]
    names += [f'cos_theta_A_{a}_times_cos_theta_B_{b}' for a in AXES for b in AXES]
    values = np.stack([obs[name] for name in names], -1)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite spin inputs')
    return values.astype(np.float32)


def evaluate(exporter, panel, draws):
    """Same selected events and w/K histograms; no inference/unfolding/refit."""
    w = np.asarray(panel['event_weight'], dtype=float)
    ids, category = np.asarray(panel['source_ids']), np.asarray(panel['category'])
    n = len(w)
    if w.shape != (n,) or ids.shape != (n,) or category.shape != (n,) or n == 0 or len(np.unique(ids)) != n:
        raise ValueError('Need aligned unique event identities, weights and categories')
    if not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
        raise ValueError('Invalid event weights')
    draws = np.asarray(draws)
    if draws.ndim != 4 or draws.shape[0] != n or draws.shape[1] < 1 or draws.shape[2:] != (2, 2):
        raise ValueError('Expected [events,K,2,2] offsets')
    va, vb = panel['visible_a'], panel['visible_b']
    target = mapped_observables(exporter, va, vb, method='target',
                               truth_tau_a=panel['truth_tau_a'], truth_tau_b=panel['truth_tau_b'])
    rows = [mapped_observables(exporter, va, vb, method='evenet', deltas=draws[:, k]) for k in range(draws.shape[1])]
    pred = {key: np.stack([row[key] for row in rows], 1) for key in target}
    kappas = analyzing_powers(category)
    channels = {}
    for ch in np.unique(category):
        take = category == ch
        if w[take].sum() <= 0:
            raise ValueError('Empty weighted channel')
        t = channel_matrices({key: v[take] for key, v in target.items()}, w[take], kappas[take][0])
        p = channel_matrices({key: v[take] for key, v in pred.items()}, w[take, None]/draws.shape[1], kappas[take][0])
        for result in (t, p):
            for histogram in result['histograms'].values():
                histogram.pop('error'); histogram.pop('sumw2')
        diff = np.asarray(p['Cij'])-np.asarray(t['Cij'])
        channels[str(int(ch))] = dict(target=t, generated=p, error=float(np.linalg.norm(diff)))
    return dict(protocol=PROTOCOL, channels=channels, unfolded=False,
        reference='Root exporter target: actual MC tau p4 + reco visible, no post-calibration',
        prediction='Root exporter evenet: offsets + reco visible followed by common.post_calibrate_tau_tau',
        limitation='Same caller-selected events only; PI deployed version, cuts, normalization and uncertainty not certified. '
                   'Target and prediction have different reconstruction maps; zero error is not guaranteed even for truth offsets.')
