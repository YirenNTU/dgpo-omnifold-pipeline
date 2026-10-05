#!/usr/bin/env python3
"""Plot Truth, Prediction, and Prediction - Truth from saved tau evaluations.

This script reads samples; it does not load a model or run inference. Checkpoint
metadata identifies the policy that produced the supplied evaluation samples.
Both axes describe the same leg. Candidate zero is the default, as in W&B.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
import numpy as np


def load_coordinates(args):
    if args.paired_coordinates:
        with np.load(args.paired_coordinates, allow_pickle=False) as f:
            ids = f['source_ids'].copy()
            truth, prediction = f['truth'].copy(), f['prediction'].copy()
            weights = f['weight'].copy()
        if args.leg != 'a' or args.candidate != 0:
            raise ValueError('Compact coordinates contain leg a, candidate zero only')
    else:
        if args.validation_panel is None:
            raise ValueError('--validation-panel is required with --measurements')
        with np.load(args.measurements, allow_pickle=False) as m, np.load(args.validation_panel, allow_pickle=False) as v:
            ids = m['source_ids'].copy()
            if not np.array_equal(ids, v['source_ids']):
                raise ValueError('Prediction and truth event IDs/order differ')
            deltas, targets = m['deltas'], v['truth_deltas']
            if deltas.ndim != 4 or deltas.shape[2:] != (2, 2) or targets.shape != (len(ids), 2, 2):
                raise ValueError('Expected deltas [N,K,2,2] and truth_deltas [N,2,2]')
            if deltas.shape[0] != len(ids) or not 0 <= args.candidate < deltas.shape[1]:
                raise ValueError('Invalid prediction event count or candidate index')
            leg = 0 if args.leg == 'a' else 1
            truth, prediction = targets[:, leg, :].copy(), deltas[:, args.candidate, leg, :].copy()
            weights = v['event_weight'].copy()
            if 'weight' in m and not np.array_equal(m['weight'], weights):
                raise ValueError('Saved prediction weights differ from validation weights')
    if len(ids) == 0 or len(np.unique(ids)) != len(ids):
        raise ValueError('Empty or duplicated event identities')
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (len(ids),) or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError('Require finite nonnegative event weights with positive total')
    series = {}
    for key, values in [('truth', truth), ('prediction', prediction)]:
        values = np.asarray(values, dtype=np.float64).copy()
        if values.shape != (len(ids), 2) or not np.isfinite(values).all():
            raise ValueError('Require finite [N,2] angular coordinates')
        values[:, 1] = np.arctan2(np.sin(values[:, 1]), np.cos(values[:, 1]))
        series[key] = values
    return series, weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--measurements', type=Path, help='Saved evaluation measurements.npz')
    inputs.add_argument('--paired-coordinates', type=Path, help='Previously exported leg-a/candidate-0 compact NPZ')
    parser.add_argument('--validation-panel', type=Path, help='Matching validation_panel.npz with truth_deltas')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--leg', choices=['a', 'b'], default='a')
    parser.add_argument('--candidate', type=int, default=0)
    parser.add_argument('--bins', type=int, default=100)
    parser.add_argument('--core-quantile', type=float, default=.995)
    parser.add_argument('--run', default='Saved evaluation')
    parser.add_argument('--policy-step', type=int, help='Run policy step; checked against saved report when available')
    parser.add_argument('--checkpoint', help='Provenance label only; samples must come from this checkpoint')
    args = parser.parse_args()
    if args.bins < 2 or not 0 < args.core_quantile < 1:
        parser.error('Require bins >= 2 and 0 < core-quantile < 1')
    if args.measurements:
        report_file = args.measurements.parent / 'report.json'
        if report_file.is_file():
            saved_step = json.loads(report_file.read_text()).get('policy_step')
            if saved_step is not None:
                if args.policy_step is not None and args.policy_step != saved_step:
                    raise ValueError('--policy-step differs from the saved evaluation report')
                args.policy_step = saved_step
    series, weights = load_coordinates(args)
    args.output.mkdir(parents=True, exist_ok=True)
    joined = np.concatenate(list(series.values()))
    extents = dict(full_range=np.maximum(np.abs(joined).max(axis=0) * 1.02, 1e-6),
                   core_zoom=np.maximum(np.quantile(np.abs(joined), args.core_quantile, axis=0) * 1.05, 1e-6))
    prefix = f'leg_{args.leg}'
    labels = dict(xlabel=rf'Leg {args.leg} target $\Delta\theta_{args.leg}$ [rad]',
                  ylabel=rf'Leg {args.leg} target $\Delta\phi_{args.leg}$ [rad]')
    policy_label = f'checkpoint step {args.policy_step}' if args.policy_step is not None else 'saved policy'
    caption = f'{args.run} | {len(weights):,} matched events | candidate {args.candidate}'
    output = dict(run=args.run, checkpoint=args.checkpoint, policy_step=args.policy_step,
                  events=len(weights), leg=args.leg, candidate=args.candidate,
                  measurements=str(args.measurements) if args.measurements else None,
                  validation_panel=str(args.validation_panel) if args.validation_panel else None,
                  paired_coordinates=str(args.paired_coordinates) if args.paired_coordinates else None,
                  normalization='Full event-weight sum and bin area; no classifier reweighting',
                  difference='Prediction density minus Truth density')
    plt.rcParams.update({'font.size': 12, 'figure.facecolor': 'white'})
    for mode, extent in extents.items():
        x, y = [np.linspace(-r, r, args.bins + 1) for r in extent]
        area = np.diff(x)[:, None] * np.diff(y)[None, :]
        density, outside = {}, {}
        for key, values in series.items():
            counts = np.histogram2d(values[:, 0], values[:, 1], bins=(x, y), weights=weights)[0]
            density[key] = counts / weights.sum() / area
            outside[key] = float(1 - counts.sum() / weights.sum())
        difference = density['prediction'] - density['truth']
        signed_mass = float(np.sum(difference * area))
        if not np.isclose(signed_mass, outside['truth'] - outside['prediction'], atol=1e-12):
            raise ValueError('Histogram subtraction mass consistency failed')
        scope = 'Full sample range' if mode == 'full_range' else 'Central zoom; full-event normalization'
        vmax = max(h.max() for h in density.values())
        nonzero = np.concatenate([h[h > 0] for h in density.values()])
        if not len(nonzero):
            raise ValueError('Plot range contains no positive-weight events')
        norm = (LogNorm(vmin=nonzero.min(), vmax=vmax) if mode == 'full_range'
                else Normalize(vmin=0, vmax=vmax))
        cmap = plt.get_cmap('magma').copy()
        cmap.set_bad('#f0f0f0')
        fig, axes = plt.subplots(1, 2, figsize=(12.3, 5.4), layout='constrained')
        for ax, key in zip(axes, ('truth', 'prediction')):
            h = density[key].T
            im = ax.pcolormesh(x, y, np.ma.masked_equal(h, 0) if mode == 'full_range' else h,
                              cmap=cmap, norm=norm, rasterized=True)
            ax.set(**labels, xlim=(x[0], x[-1]), ylim=(y[0], y[-1]),
                   title='Truth' if key == 'truth' else f'Prediction | {policy_label}')
            ax.text(.025, .975, f'Outside view: {outside[key]:.2%}', transform=ax.transAxes,
                    va='top', color='white', fontsize=10,
                    bbox=dict(facecolor='black', alpha=.55, edgecolor='none', pad=3))
        fig.colorbar(im, ax=axes, label='Probability density [rad$^{-2}$]' + (' | log scale' if mode == 'full_range' else ''))
        fig.suptitle(caption + '\n' + scope, fontsize=14)
        save_figure(fig, args.output / f'{prefix}_joint_{mode}')
        limit = max(float(np.abs(difference).max()), 1e-12)
        fig, ax = plt.subplots(figsize=(8.2, 6.5), layout='constrained')
        im = ax.pcolormesh(x, y, difference.T, cmap='RdBu_r',
                          norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), rasterized=True)
        ax.set(**labels, xlim=(x[0], x[-1]), ylim=(y[0], y[-1]),
               title=f'Prediction − Truth | {policy_label}')
        fig.colorbar(im, ax=ax, label='Density difference [rad$^{-2}$]')
        fig.suptitle(caption + '\n' + scope, fontsize=13)
        ax.text(.02, .98, f'Outside view: Truth {outside["truth"]:.2%}, Prediction {outside["prediction"]:.2%}',
                transform=ax.transAxes, va='top', fontsize=10,
                bbox=dict(facecolor='white', edgecolor='none', alpha=.8))
        save_figure(fig, args.output / f'{prefix}_difference_{mode}')
        output[mode] = dict(x_edges=x.tolist(), y_edges=y.tolist(), outside_view=outside,
                            density={k: h.tolist() for k, h in density.items()},
                            density_difference=difference.tolist(), signed_mass_difference=signed_mass)
    (args.output / 'joint_histograms.json').write_text(json.dumps(output, indent=2) + '\n')
    print(f'Saved Truth/Prediction and Prediction−Truth PNG/PDF plots to {args.output}')


def save_figure(fig, path):
    fig.savefig(path.with_suffix('.png'), dpi=180, bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
