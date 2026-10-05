"""Fixed-baseline 1D monitoring; candidate zero, never reward-selected samples."""
import json
from pathlib import Path

import numpy as np
import torch

from RL.DGPO_neutrino.domains.ztautau import (
    reconstruct_tau_angles_from_deltas, tau_back_to_back_metrics,
)


def observables(panel, delta):
    delta = torch.as_tensor(np.asarray(delta), dtype=torch.float64).clone()
    delta[..., 1] = torch.atan2(delta[..., 1].sin(), delta[..., 1].cos())
    batch = {f'lead_{leg}_visible_{key}': panel['visible_'+leg][:, j]
             for leg in ('a', 'b') for j, key in enumerate(('E', 'px', 'py', 'pz'))}
    angles = reconstruct_tau_angles_from_deltas(delta[None], batch)
    out = {}
    for i, leg in enumerate(('a', 'b')):
        for j, coord in enumerate(('theta', 'phi')):
            out[f'target/tau_{leg}_delta_{coord}'] = delta[:, i, j].numpy()
            out[f'reco/tau_{leg}_{coord}'] = angles[i][0, :, j].numpy()
    for name, value in tau_back_to_back_metrics(delta[None], batch).items():
        if name != 'back_to_back_loss':
            out['topology/'+name] = value[0].numpy()
    return out


def probability(values, weights, edges):
    """Include explicit under/overflow bins: no hidden tail truncation."""
    values, weights = np.asarray(values), np.asarray(weights, dtype=float)
    if (not np.isfinite(values).all() or not np.isfinite(weights).all()
            or (weights < 0).any() or weights.sum() <= 0):
        raise ValueError('Invalid marginal values/event weights')
    bins = np.r_[-np.inf, edges, np.inf]
    return np.histogram(values, bins=bins, weights=weights)[0]/weights.sum()


def distances(p, q):
    m = (p+q)/2
    def kl(x):
        keep = x > 0
        return np.sum(x[keep]*np.log(x[keep]/m[keep]))
    return {'tv': float(np.abs(p-q).sum()/2), 'jsd': float((kl(p)+kl(q))/2)}


def monitor(panel, current, baseline_file, folder):
    """Return scalar logs and saved plots; initial baseline never follows refits."""
    folder, baseline_file = Path(folder), Path(baseline_file)
    if not baseline_file.is_file():
        return {'tau/marginal/baseline_available': 0}, {}
    with np.load(baseline_file, allow_pickle=False) as f:
        if not np.array_equal(f['source_ids'], panel['source_ids']):
            raise ValueError('Step-zero marginal baseline event IDs differ')
        baseline = f['deltas'][:, 0].copy()
    series = dict(truth=observables(panel, panel['truth_deltas']),
                  initial=observables(panel, baseline), current=observables(panel, current[:, 0]))
    w = panel['event_weight']
    out = {'tau/marginal/baseline_available': 1, 'tau/marginal/events': len(w)}
    records = {}
    for name in series['truth']:
        # Fixed truth + initial ranges across all epochs, independent of current model.
        fixed = np.concatenate([series[x][name] for x in ('truth', 'initial')])
        if name.endswith('_phi'):
            lo, hi = -np.pi, np.pi
        elif name.startswith('reco/'):
            lo, hi = 0., np.pi
        else:
            lo, hi = np.quantile(fixed, [.005, .995])
            pad = max((hi-lo)*.05, 1e-6)
            lo, hi = lo-pad, hi+pad
        edges = np.linspace(lo, hi, 61)
        probs = {key: probability(value[name], w, edges) for key, value in series.items()}
        for key in ('initial', 'current'):
            for metric, value in distances(probs[key], probs['truth']).items():
                out[f'tau/marginal/{metric}/{key}/{name}'] = value
        for metric in ('tv', 'jsd'):
            out[f'tau/marginal/{metric}/change/{name}'] = (
                out[f'tau/marginal/{metric}/current/{name}']-out[f'tau/marginal/{metric}/initial/{name}'])
        for key, p in probs.items():
            out[f'tau/marginal/tail_mass/{key}/{name}'] = float(p[0]+p[-1])
        records[name] = dict(edges=edges.tolist(), probabilities={k: p.tolist() for k, p in probs.items()})
    (folder/'marginals.json').write_text(json.dumps(dict(
        baseline=str(baseline_file), metrics=out, histograms=records,
        scope='Candidate zero per condition; event weights only, no classifier weights. '
              'JSD is natural-log divergence, TV includes under/overflow. '
              'Ranges fixed by truth and step zero; no uncertainty estimate.'), indent=2)+'\n')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    paths = {}
    for group in ('target', 'reco', 'topology'):
        names = [n for n in records if n.startswith(group+'/')]
        fig = Figure(figsize=(11, 3.5*((len(names)+1)//2)), layout='constrained')
        FigureCanvasAgg(fig)
        axes = fig.subplots((len(names)+1)//2, 2, squeeze=False).ravel()
        for ax, name in zip(axes, names):
            record = records[name]
            edges = np.array(record['edges'])
            for key, label in [('truth', 'Truth'), ('initial', 'Initial step 0'), ('current', 'Current')]:
                p = np.array(record['probabilities'][key])
                ax.stairs(p[1:-1]/np.diff(edges), edges, label=f'{label} (outside {p[0]+p[-1]:.2%})')
            ax.set_title(name.split('/')[-1]); ax.set_ylabel('Probability density')
            ax.set_xlabel('cosine' if name.endswith('cos_opening') else 'radians')
            ax.legend(fontsize=8)
        fig.suptitle('Native samples | one draw/condition | no ratio reweighting')
        path = folder/f'marginals_{group}.png'
        fig.savefig(path, dpi=130)
        paths[group] = path
        fig.clear()
    return out, paths
