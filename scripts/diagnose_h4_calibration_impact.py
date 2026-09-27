#!/usr/bin/env python3
"""Production calibration impact on saved matched candidates, NOT QI closure.

No checkpoint loading, fitting, event removal or candidate selection. The saved
angular target is a direction reference, not a replacement for truth tau p4.
"""
import argparse
import json
from pathlib import Path
import sys

import awkward as ak
import numpy as np
import torch
import vector

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import post_calibrate_tau_tau, CM_ENERGY, TAU_MASS
from diagnose_h4_ratio_tail import unpack

vector.register_awkward()
ARM_NAMES = ('pretrained10pct', 'step1110', 'pretrainedfull')


def separation(a, b):
    return np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.sum(a*b, axis=-1))


def directions(fields, deltas):
    z = np.asarray(deltas, dtype=np.float64)
    if z.ndim != 3 or z.shape[-1] != 4 or not np.isfinite(z).all():
        raise ValueError('Expected finite (events,K,4) deltas')
    result = []
    for j, leg in enumerate(('a', 'b')):
        p = np.stack([fields[f'lead_{leg}_visible_{axis}'].numpy().reshape(-1) for axis in ('px','py','pz')], -1).astype(np.float64)
        if not np.isfinite(p).all() or (np.linalg.norm(p, axis=-1) == 0).any():
            raise ValueError('Invalid visible momentum')
        theta = np.arctan2(np.hypot(p[:,0], p[:,1]), p[:,2])[:,None] + z[:,:,2*j]
        phi = np.arctan2(p[:,1], p[:,0])[:,None] + z[:,:,2*j+1]
        if ((theta < 0) | (theta > np.pi)).any():
            raise ValueError('Theta outside physical range; do not silently drop events')
        result.append(np.stack([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)], -1))
    return np.stack(result, -2)


def calibrate(u):
    """Call the actual production function, including its fixed E/m conventions."""
    # These are the pre-calibration values used in export_evenet_qi_inputs.py.
    momentum = np.sqrt((91.2/2)**2 - 1.777**2)
    flat = u.reshape(-1, 2, 3)
    p4 = [ak.zip(dict(px=flat[:,j,0]*momentum, py=flat[:,j,1]*momentum,
                     pz=flat[:,j,2]*momentum, energy=np.full(len(flat),91.2/2)),
                 with_name='Momentum4D') for j in range(2)]
    a,b = post_calibrate_tau_tau(*p4)
    out = []
    for v in (a,b):
        xyz = np.stack([ak.to_numpy(v.px),ak.to_numpy(v.py),ak.to_numpy(v.pz)], -1)
        out.append(xyz/np.linalg.norm(xyz,axis=-1,keepdims=True))
    result = np.stack(out, -2).reshape(u.shape)
    if not np.isfinite(result).all():
        raise ValueError('Production calibration produced nonfinite values')
    return result


def distribution_summary(x):
    x = np.asarray(x)
    means = x.mean(1)
    return dict(mean=float(x.mean()), median=float(np.median(x)),
                q90=float(np.quantile(x,.9)), q99=float(np.quantile(x,.99)),
                event_cluster_se=float(means.std(ddof=1)/np.sqrt(len(means))))


def check_panel(reference, other):
    if reference['packing_spec'] != other['packing_spec']:
        raise ValueError('Packing specs differ')
    for key in ('condition','truth','test_rows','pool_rows'):
        if not torch.equal(reference[key], other[key]):
            raise ValueError(f'Panels differ: {key}')


def load_arm(path):
    path = Path(path)
    if not (path/'COMPLETE').is_file():
        raise ValueError(f'Incomplete coverage arm: {path}')
    manifest = json.loads((path/'manifest.json').read_text())
    panel = torch.load(path/'panel.pt', map_location='cpu', weights_only=True)
    candidates = torch.load(path/'candidates.pt', map_location='cpu', weights_only=True)['generated'].numpy()
    if candidates.shape != (len(panel['truth']),128,4) or len(panel['truth']) < 2:
        raise ValueError('Invalid candidate shape or panel size')
    return manifest,panel,candidates


def analyze(panel, candidates):
    fields = unpack(dict(test_condition=panel['condition'],packing_spec=panel['packing_spec']))
    target = directions(fields,panel['truth'].numpy().reshape(-1,1,4))
    projected_target = calibrate(target)
    # Compare both stages to the same unmodified angular target. The projected
    # reference is additional, clearly labelled; it never replaces the target.
    target_projection_shift = separation(target,projected_target).mean(-1)
    report = dict(target_projection_shift_radians=distribution_summary(target_projection_shift), arms={})
    event_errors = {}
    for name,z in candidates.items():
        before = directions(fields,z)
        after = calibrate(before)
        result = {'calibration_shift_radians': distribution_summary(separation(before,after).mean(-1))}
        for stage,u in [('before',before),('after',after)]:
            errors = separation(u,target).mean(-1)
            result[stage] = dict(
                opening_deficit_radians=distribution_summary(np.pi-separation(u[:,:,0],u[:,:,1])),
                mean_leg_error_to_saved_target_radians=distribution_summary(errors),
                mean_leg_error_to_projected_target_radians=distribution_summary(separation(u,projected_target).mean(-1)),
                direction_mean_bias_xyz=(u.mean((0,1))-target.mean((0,1))).tolist())
            event_errors[f'{name}/{stage}'] = errors.mean(1)
        report['arms'][name] = result
    report['paired_after_minus_before'] = {}
    for name in candidates:
        d = event_errors[name+'/after']-event_errors[name+'/before']
        report['paired_after_minus_before'][name] = dict(mean_error_change_radians=float(d.mean()),
            paired_event_se=float(d.std(ddof=1)/np.sqrt(len(d))))
    return report, event_errors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ARM_NAMES:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('Choose a fresh output directory')
    manifests, candidates = {}, {}
    panel = None
    for name in ARM_NAMES:
        meta, observed, z = load_arm(getattr(args,name))
        if panel is None:
            panel = observed
        else:
            check_panel(panel,observed)
            for key in ('seed','workers','batch_size','ddim_steps','K','events'):
                if meta[key] != manifests[ARM_NAMES[0]][key]:
                    raise ValueError(f'Unmatched sampling setting: {key}')
        if meta.get('arm', 'step1110') != name or meta['policy_updates'] != 0:
            raise ValueError('Wrong arm or policy updates')
        manifests[name] = meta
        candidates[name] = z
    report, errors = analyze(panel,candidates)
    report.update(schema='h4-calibration-impact-v1', events=len(panel['truth']), candidates_per_event=128,
        provenance=manifests, production_calibration=dict(CM_ENERGY=CM_ENERGY,TAU_MASS=TAU_MASS),
        physics_closure_complete=False,
        limitations=['No B_i/C_ij, response matrix or unfolding result: source truth p4, channel, weights and QIProcessor are required.',
                     'Saved truth deltas define a direction reference, not the full truth tau four-vector.',
                     'Post-calibration back-to-back closure is imposed by construction, not learned.',
                     'Per-event angular error is a diagnostic, not a proper scoring rule for a generative distribution.',
                     'All 128 candidates have equal weight; uncertainty clusters by event, not candidate.',
                     'Reused panel and unverified pretraining overlap; no held-out-generalization claim.'])
    args.output.mkdir(parents=True,exist_ok=False)
    with (args.output/'report.json').open('x') as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
    np.savez_compressed(args.output/'event_direction_errors.npz', **errors)
    (args.output/'COMPLETE').write_text('h4-calibration-impact-v1\n')
    print(json.dumps({k:v for k,v in report.items() if k!='provenance'},indent=2,allow_nan=False))
    print('REPORT:',args.output/'report.json')


if __name__ == '__main__':
    main()
