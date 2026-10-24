"""Cached mixture test calling EveNet's original nu2flow functions unchanged.

The original interface has one prediction per event. A single seeded categorical
draw realizes each candidate distribution; it is not a fractional-histogram error
calculation. Original same-sample closure is descriptive, not a coverage test.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.ttbar_general_classifier.run import parts, read_config, write_json
from experiments.ttbar_general_classifier.core import (
    GeneralClassifier, mixture, top_oracle_inputs, p4_from_columns,
)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def choose(weights, uniforms):
    weights = np.asarray(weights, float)
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('Invalid candidate probabilities')
    np.testing.assert_allclose(weights.sum(1), 1, atol=1e-6)
    cdf = weights.cumsum(1)
    cdf[:, -1] = 1.
    return (uniforms[:, None] >= cdf).sum(1)


def load_scored(c, out):
    status = json.loads((out/'FIT_COMPLETE.json').read_text())
    if not status['plateau']:
        raise ValueError('Classifier fit did not reach its plateau criterion')
    payload = torch.load(out/'classifier-best.pt', map_location='cpu', weights_only=True)
    stats = payload['stats']
    if stats['oracle_reference'] != c['mixture']['oracle']:
        raise ValueError('Oracle reference differs from training normalization')
    model = GeneralClassifier(stats['condition_dim'], stats['candidate_dim'], **c['head'])
    model.load_state_dict(payload['state_dict']); model.eval()
    arrays = {}
    with torch.no_grad():
        for _, _, a in parts(out, 'validation'):
            keep = (a['partition'] != 0) & a['physics_keep'].astype(bool)
            if not keep.any():
                continue
            cond = torch.tensor(a['condition'][keep])
            labels = torch.tensor(a['labels'][keep])
            features = a['generated_features'][keep]
            logits = np.stack([model(cond, (torch.tensor(features[:, k])-stats['mean'])/
                                    stats['scale'], labels).numpy()
                               for k in range(features.shape[1])], 1)
            arrays.setdefault('logits', []).append(logits)
            for key in ('ids', 'truth', 'candidates', 'partition'):
                arrays.setdefault(key, []).append(np.asarray(a[key][keep]))
            for key in a:
                if key.startswith('EXTRA/'):
                    arrays.setdefault(key, []).append(np.asarray(a[key][keep]))
    return {k: np.concatenate(v) for k, v in arrays.items()}, stats


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--evenet-root', type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import vector
    from threadpoolctl import threadpool_limits
    threadpool_limits(2); torch.set_num_threads(2)
    c = read_config(args.config)
    out = args.output.resolve(); dest = out/'native_nu2flow'
    if dest.exists():
        raise ValueError(f'Preserve existing output: {dest}')
    eve = args.evenet_root.resolve()
    sources = {key: eve/path for key, path in {
        'observables': 'downstreams/nu2flow/distributions.py',
        'solver': 'downstreams/nu2flow/unfolding/unfold.py',
        'plotter': 'downstreams/plotting/unfolding.py',
    }.items()}
    sys.path.insert(0, str(eve))
    distributions = load_module('native_distributions', sources['observables'])
    native = load_module('native_unfold', sources['solver'])
    native.ROOT.gROOT.SetBatch(True)
    plotter = load_module('native_unfold_plot', sources['plotter'])
    arrays, stats = load_scored(c, out)
    extra = {k: v for k, v in arrays.items() if k.startswith('EXTRA/')}
    tops, target = top_oracle_inputs(arrays['candidates'], extra, arrays['truth'], c['mixture']['oracle'])
    general, mixed, _ = mixture(arrays['logits'], tops, target, stats['oracle_scale'],
                                alpha=c['mixture']['alpha'], temperature=c['mixture']['temperature'])
    probabilities = [np.full_like(general, 1/general.shape[1]), general, mixed]
    names = ['Raw diffusion', 'General', 'General70 + oracle30']
    # Common random numbers, stable under shard/order changes, one seed only.
    uniforms = np.array([(int.from_bytes(hashlib.blake2b(
        f"nu2flow-choice:{c['seed']}:{event}".encode(), digest_size=8).digest(), 'little') >> 11)
        / 2**53 for event in arrays['ids']])
    selected = [choose(w, uniforms) for w in probabilities]
    def vec(p):
        return vector.array(dict(E=p[..., 0], px=p[..., 1], py=p[..., 2], pz=p[..., 3]))
    shared = dict(
        truth_top=vec(np.stack([p4_from_columns(extra, f'EXTRA/truth_t{i}/t', True) for i in (1, 2)], 1)),
        truth_lepton=vec(np.stack([p4_from_columns(extra, f'EXTRA/truth_t{i}/l', True) for i in (1, 2)], 1)),
        lepton=vec(np.stack([p4_from_columns(extra, f'EXTRA/t{i}/l') for i in (1, 2)], 1)),
        plot_truth_top=vec(target),
    )
    dest.mkdir()
    np.savez_compressed(dest/'candidate-selection.npz', ids=arrays['ids'], partition=arrays['partition'],
                        uniform=uniforms, raw=selected[0], general=selected[1], mixture=selected[2])
    result = dict(
        sources={k: dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for k, p in sources.items()},
        config=c,
        source_checkpoint=json.loads((out/'cache_manifest.json').read_text())['source_checkpoint'],
        classifier_sha256=hashlib.sha256((out/'classifier-best.pt').read_bytes()).hexdigest(),
        oracle_scale=stats['oracle_scale'],
        events=len(uniforms), seed=c['seed'], kreg=7, bins=16,
        error_method='Original RooUnfoldSvd(response, h_measure, 7).Hunfold(2); default data covariance; no response-MC systematics enabled',
        plotted_metric='sigma(prednu unfolded counts) / sigma(truthnu unfolded counts)',
        candidate_adapter='One categorical draw per event; shared ID-keyed uniform across three arms; no generation',
        selection='Existing cache physics_keep AND partition!=0, common across all arms; not original notebook BASE_SELECTION',
        original_selection=distributions.BASE_SELECTION,
        selection_limit='Cache uses compute_base_recon_cut AND both assignment masks; no full point cloud/num_bjet in downloaded candidate cache to reproduce notebook selection',
        dphi='Original signed vector.deltaphi/pi with original nonnegative bins; invalid bins/misses unchanged',
        closure_limit='Original main uses the same events for response and measured histogram; closure is not independent coverage evidence',
        targets={}, validity={},
    )
    frames = []
    for j, (name, index) in enumerate(zip(names, selected)):
        df = distributions.preprocess(dict(shared, top=vec(tops[np.arange(len(index)), index])))
        df.to_csv(dest/f'arm{j}-input.csv', index=False)
        binned = native.pre_bin(df, native.hist_setup())
        result['validity'][name] = {
            var: dict(truth_valid=int(binned[f'{var}_truth_final'].notna().sum()),
                      reco_valid=int(binned[f'{var}_prednu_final'].notna().sum()),
                      misses=int((binned[f'{var}_truth_final'].notna() & binned[f'{var}_prednu_final'].isna()).sum()),
                      reco_only_not_modelled_by_original_response=int((binned[f'{var}_truth_final'].isna() & binned[f'{var}_prednu_final'].notna()).sum()))
            for var in native.hist_setup() if var != 'm_tt'}
        frame = native.main(df)
        frame.to_csv(dest/f'arm{j}-unfolded.csv', index=False)
        frames.append(frame)
        print(f'Native nu2flow complete: {name}', flush=True)
    labels = {
        'dphi_ll': ([r'0–0.25', r'0.25–0.5', r'0.5–0.75', r'0.75–1'], r'$\Delta\phi_{\ell\ell}/\pi$'),
        'pt_t1': (['<75', '75–125', '125–175', '≥175'], r'$p_T^t$ [GeV]'),
        'pt_t2': (['<75', '75–125', '125–175', '≥175'], r'$p_T^{\bar t}$ [GeV]'),
        'pt_tt': (['<70', '70–140', '140–200', '≥200'], r'$p_T^{t\bar t}$ [GeV]'),
        'y_tt': (['<−1', '−1–0', '0–1', '≥1'], r'$y_{t\bar t}$'),
    }
    # Keep labels local; mathematical computation and plotting helper are upstream.
    colors = ['#0072B2', '#E69F00', '#009E73']
    for var, (ticks, xlabel) in labels.items():
        denominator = frames[0][f'{var}_truthnu_unfold_error'].to_numpy()
        if not np.isfinite(denominator).all() or (denominator <= 0).any():
            raise ValueError(f'Invalid native truth-neutrino denominator: {var}')
        methods = []; rows = []
        for j, (name, frame) in enumerate(zip(names, frames)):
            np.testing.assert_allclose(frame[f'{var}_truthnu_unfold_error'], denominator, rtol=1e-10)
            sigma = frame[f'{var}_prednu_unfold_error'].to_numpy()
            value = frame[f'{var}_prednu_unfold_content'].to_numpy()
            truth = frame[f'{var}_prednu_truth_content'].to_numpy()
            if not np.isfinite(sigma).all() or (sigma <= 0).any():
                raise ValueError(f'Invalid native predicted-neutrino uncertainty: {var}/{name}')
            rows.append(dict(name=name, sigma=sigma.tolist(), truthnu_sigma=denominator.tolist(),
                             uncertainty_ratio=(sigma/denominator).tolist(), count_sigma_rss=float(np.linalg.norm(sigma)),
                             unfolded=value.tolist(), truth=truth.tolist(),
                             same_sample_max_abs_residual_over_fit_sigma=float(np.max(np.abs(value-truth)/sigma))))
            methods.append(dict(name=name, color=colors[j], data=sigma/denominator))
        result['targets'][var] = rows
        plotter.plot_uncertainty_with_ratio(
            [r'$m_{t\bar t}<400$', '400–500', '500–800', r'$m_{t\bar t}\geq800$'],
            ticks, xlabel, methods, names[0], dest, save_name=f'{var}.png',
            ratio_baseline_min=-.5, ratio_baseline_max=.5, ratio_y_label='Improvement to raw')
    write_json(dest/'report.json', result)
    html = ['<html><meta charset="utf-8"><title>TTbar frozen classifier — nu2flow</title>',
            '<style>body{font:16px system-ui;max-width:1200px;margin:30px auto}img{width:100%}</style>',
            '<h1>Frozen General + top oracle</h1>',
            '<p>Original nu2flow functions: 4×4 bins, SVD kreg=7. Upper panel: predicted-neutrino / truth-neutrino unfolded count uncertainty. Lower panel: improvement to raw.</p>',
            '<p>One candidate drawn per event, seed 42. Existing classifier-cache selection; this differs from the notebook population. Original signed dilepton Δφ and same-sample response are preserved. Response-MC uncertainty is not included in these original errors.</p>',
            '<p><a href="report.json">Definitions and source hashes</a> · <a href="../evaluation/index.html">Separate all-candidate weighted diagnostics</a></p>']
    for var in labels:
        html.append(f'<h2>{var}</h2><img src="uncertainty/{var}.png">')
    (dest/'index.html').write_text('\n'.join(html))


if __name__ == '__main__':
    main()
