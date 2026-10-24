"""Root export_evenet_qi_inputs target/evenet geometry, without ROOT.

Target: complete MC tau four-vectors and reconstructed visible four-vectors.
Prediction: angular offsets and reconstructed visible, then common calibration.
Only the target path accepts truth p4. Both feed the same observable estimator.
The legacy util/post_calibration protocol remains separate for old checkpoints.
"""
from pathlib import Path

import awkward as ak
import numpy as np
import vector

from scripts.tau_exporter_cij import PROTOCOL
from scripts.tau_tihsu_cij import observables, from_deltas, PROTOCOL as LEGACY_PROTOCOL


def panel_protocol(panel):
    protocol = str(panel.get('spin_protocol', LEGACY_PROTOCOL))
    if protocol not in (LEGACY_PROTOCOL, PROTOCOL):
        raise ValueError('Unknown Spin reconstruction protocol')
    return protocol


def target_spin_observables(panel, indices=slice(None)):
    if panel_protocol(panel) == PROTOCOL:
        if not all(k in panel for k in ('truth_tau_a', 'truth_tau_b')):
            raise ValueError('Root target requires complete truth tau p4, not just offsets')
        return target_observables(*(panel[k][indices] for k in
            ('visible_a', 'visible_b', 'truth_tau_a', 'truth_tau_b')))
    return from_deltas(*(panel[k][indices] for k in ('visible_a', 'visible_b', 'truth_deltas')))


def prediction_spin_observables(panel, draws, indices=slice(None)):
    va, vb = panel['visible_a'][indices], panel['visible_b'][indices]
    if draws.ndim == 4:
        va, vb = va[:, None], vb[:, None]
    if panel_protocol(panel) == PROTOCOL:
        return prediction_observables(va, vb, draws)
    return from_deltas(va, vb, draws)


def p4(values, name):
    values = np.asarray(values, dtype=np.float64)
    if values.shape[-1:] != (4,) or not np.isfinite(values).all():
        raise ValueError(name+' requires finite p4 in E,px,py,pz order')
    return values


def prediction_tau_pair(visible_a, visible_b, deltas):
    """Root map on its valid domain; periodic Cartesian extension outside it.

    A finite direction sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)
    remains defined when a generated theta crosses a pole. The original
    pt/theta representation instead produces NaNs. Never clip angles or drop
    those candidates: apply the same fixed-energy opposite-direction map to
    their Cartesian directions. Normal-domain inputs retain the exact exporter
    arithmetic. This extension is explicitly versioned in the Spin protocol.
    """
    import common
    if Path(common.__file__).resolve() != Path(__file__).resolve().parents[1]/'common.py':
        raise ValueError('Root common.py required for exporter calibration')
    if (common.CM_ENERGY, common.TAU_MASS) != (91.25, 1.77):
        raise ValueError('Root calibration changed; revalidate/version the feature protocol')
    d = np.asarray(deltas, dtype=np.float64)
    if d.shape[-2:] != (2, 2) or not np.isfinite(d).all():
        raise ValueError('Expected finite (...,leg,theta/phi) offsets')
    shape = d.shape[:-2]
    vector.register_awkward()
    taus = []
    directions = []
    extended = np.zeros(int(np.prod(shape)), dtype=bool)
    for j, values in enumerate((visible_a, visible_b)):
        v = np.broadcast_to(p4(values, 'visible'), shape+(4,)).reshape(-1, 4)
        vis = ak.zip(dict(zip(('E', 'px', 'py', 'pz'), v.T)), with_name='Momentum4D')
        theta = vis.theta + d[..., j, 0].ravel()
        phi = (vis.phi + d[..., j, 1].ravel() + np.pi) % (2*np.pi) - np.pi
        theta_np, phi_np = ak.to_numpy(theta), ak.to_numpy(phi)
        extended |= (theta_np <= 0) | (theta_np >= np.pi)
        directions.append(np.stack((np.sin(theta_np)*np.cos(phi_np),
                                    np.sin(theta_np)*np.sin(phi_np), np.cos(theta_np)), -1))
        # These are the *initial* constants in root evenet_tau_pair. The root
        # common function supplies the final E=45.625 GeV and m=1.77 GeV.
        momentum = np.sqrt(45.6**2-1.777**2)
        taus.append(ak.zip(dict(pt=momentum*np.sin(theta), theta=theta, phi=phi,
                               m=ak.ones_like(theta)*1.777), with_name='Momentum4D'))
    with np.errstate(invalid='ignore', divide='ignore'):
        calibrated = common.post_calibrate_tau_tau(*taus)
        result = [np.stack([ak.to_numpy(getattr(t, k)) for k in ('E','px','py','pz')], -1)
                  for t in calibrated]
    if extended.any():
        direction = directions[0][extended] - directions[1][extended]
        length = np.linalg.norm(direction, axis=-1, keepdims=True)
        if np.any(length < 1e-12):
            raise ValueError('Undefined back-to-back axis: generated tau directions coincide')
        momentum = np.sqrt((common.CM_ENERGY/2)**2-common.TAU_MASS**2) * direction/length
        for leg, sign in enumerate((1, -1)):
            result[leg][extended, 0] = common.CM_ENERGY/2
            result[leg][extended, 1:] = sign*momentum
    return tuple(p4(t.reshape(shape+(4,)), 'calibrated tau') for t in result)


def angular_domain_report(visible_a, visible_b, deltas):
    """Count finite generated directions requiring the versioned extension."""
    d = np.asarray(deltas, dtype=np.float64)
    if d.shape[-2:] != (2, 2) or not np.isfinite(d).all():
        raise ValueError('Invalid offsets for angular-domain audit')
    masks = []
    for leg, values in enumerate((visible_a, visible_b)):
        v = np.broadcast_to(p4(values, 'visible'), d.shape[:-2]+(4,))
        theta = np.arctan2(np.linalg.norm(v[..., 1:3], axis=-1), v[..., 3])+d[..., leg, 0]
        masks.append((theta <= 0) | (theta >= np.pi))
    any_leg = masks[0] | masks[1]
    return dict(candidates=int(any_leg.size), extended_candidates=int(any_leg.sum()),
                extended_legs=[int(m.sum()) for m in masks], dropped_events=0,
                map='Cartesian periodic continuation; no angular clipping')


def prediction_observables(visible_a, visible_b, deltas):
    return observables(*prediction_tau_pair(visible_a, visible_b, deltas), visible_a, visible_b)


def target_observables(visible_a, visible_b, truth_tau_a, truth_tau_b):
    va, vb, ta, tb = [p4(v, name) for v, name in (
        (visible_a, 'visible_a'), (visible_b, 'visible_b'),
        (truth_tau_a, 'truth_tau_a'), (truth_tau_b, 'truth_tau_b'))]
    if not (va.shape == vb.shape == ta.shape == tb.shape):
        raise ValueError('Target tau/visible p4 must be event-aligned')
    return observables(ta, tb, va, vb)


def attach_truth_p4(panel, directory):
    """Join full truth tau by composite event ID from verified filtered parquet.

    Never infer its energy from offsets, missing momentum, or a beam constraint.
    MC visible is deliberately not read here; the classifier uses reco visible.
    """
    from evenet.dataset.filtered_data import validate_filtered_dataset
    from scripts.diagnose_ztautau_cij import read_event_table, source_ids, align, SOURCE_ID_COLUMNS
    validate_filtered_dataset(directory)
    prefixes = ('truth_tau_a', 'truth_tau_b', 'lead_a_visible', 'lead_b_visible')
    columns = list(SOURCE_ID_COLUMNS)+['event_weight', 'event_category']
    columns += [name+'_'+k for name in prefixes for k in ('E','px','py','pz')]
    table = read_event_table(directory, columns)
    data = {k: np.asarray(table[k].to_numpy()) for k in columns}
    order = align(source_ids(data), panel['source_ids'])
    data = {k: v[order] for k, v in data.items()}
    np.testing.assert_array_equal(data['event_weight'], panel['event_weight'])
    np.testing.assert_array_equal(data['event_category'], panel['category'])
    ta, tb, va, vb = [np.stack([data[name+'_'+k] for k in ('E','px','py','pz')], -1)
                       .astype(np.float64) for name in prefixes]
    np.testing.assert_array_equal(va, panel['visible_a'])
    np.testing.assert_array_equal(vb, panel['visible_b'])
    # Check leg ownership and angle normalization as well as row identities.
    expected = []
    for t, v in ((ta, va), (tb, vb)):
        theta = lambda x: np.arctan2(np.linalg.norm(x[:,1:3], axis=1), x[:,3])
        phi = lambda x: np.arctan2(x[:,2], x[:,1])
        expected.append(np.stack((theta(t)-theta(v), phi(t)-phi(v)), -1))
    difference = panel['truth_deltas']-np.stack(expected, 1)
    difference[...,1] = np.arctan2(np.sin(difference[...,1]), np.cos(difference[...,1]))
    if not np.isfinite(difference).all() or np.max(np.abs(difference)) > 1e-4:
        raise ValueError('Truth tau directions differ from cached labels')
    target_observables(va, vb, ta, tb)  # Fail on unusable rows; never drop silently.
    return dict(panel, truth_tau_a=ta, truth_tau_b=tb, spin_protocol=np.array(PROTOCOL))
