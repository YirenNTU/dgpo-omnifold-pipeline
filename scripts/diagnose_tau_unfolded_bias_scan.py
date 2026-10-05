"""Paired Cij injection scan with fixed large nominal response and MC bootstraps."""
import argparse
import json
from pathlib import Path
import sys
import uuid

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'evenet_dgpo')]
import numpy as np
import yaml
from scripts.tau_bias_sampling import AXES, target_probabilities
from scripts.tau_unfold_core import (
    SVDResponse, indices, moment, response_edges, summarize_pseudoexperiments,
)


def evaluate_case(estimate, reco, coefficients, z, probability, n, repeats,
                  response_repeats, seed):
    """estimate(counts, replica): replica 0 nominal, 1..R response bootstraps.

    Resample unit-count events, not weighted histograms with sqrt(sum weights)
    errors. Response-MC spread is evaluated on the expected measured histogram.
    Joint coverage below is an empirical bootstrap approximation, not a claim
    about arbitrary detector/model systematics or unconditional coverage.
    """
    probability = np.asarray(probability, float)
    if (repeats < 2 or response_repeats < 2 or n <= 0
            or not np.isfinite(probability).all() or (probability < 0).any()
            or not np.isclose(probability.sum(), 1)):
        raise ValueError('Invalid pseudoexperiment settings')
    size = len(coefficients)
    expected_counts = n * np.bincount(reco, weights=probability, minlength=size)
    expected, stat_sigma = estimate(expected_counts, 0)
    boot = np.array([estimate(expected_counts, r)[0]
                     for r in range(1, response_repeats + 1)])
    if not np.isfinite(boot).all():
        raise ValueError('Nonfinite response bootstrap; no replicas may be dropped')
    response_sigma = float(boot.std(ddof=1))
    rng = np.random.default_rng(seed)
    # Independent RNG: arm changes cannot change pseudo-data or replica choices.
    replica_ids = np.random.default_rng(seed + 1).integers(
        1, response_repeats + 1, size=repeats)
    fixed, joint, sampled_truth = [], [], []
    for replica in replica_ids:
        counts = rng.poisson(n * probability)
        if counts.sum() == 0:
            raise ValueError('Empty pseudo-data')
        measured = np.bincount(reco, weights=counts, minlength=size)
        fixed.append(estimate(measured, 0))
        value, sigma = estimate(measured, int(replica))
        joint.append((value, np.hypot(sigma, response_sigma)))
        sampled_truth.append(float(counts @ z / counts.sum()))
    fixed, joint = np.asarray(fixed), np.asarray(joint)
    truth = float(probability @ z)
    stat = summarize_pseudoexperiments(fixed[:, 0], fixed[:, 1], truth, expected)
    combined = summarize_pseudoexperiments(joint[:, 0], joint[:, 1], truth, float(boot.mean()))
    row = dict(truth_exact=truth, expected_unfolded=float(expected),
               expected_bias=float(expected - truth), expected_stat_sigma=float(stat_sigma),
               response_mc_sigma=response_sigma,
               expected_combined_sigma=float(np.hypot(stat_sigma, response_sigma)),
               response_bootstrap_mean=float(boot.mean()),
               response_bootstrap_shift=float(boot.mean() - expected),
               sampling_ess=float(1 / (probability @ probability)),
               sampling_max_probability=float(probability.max()))
    row.update({'stat_' + key: value for key, value in stat.items()})
    row.update({'joint_bootstrap_' + key: value for key, value in combined.items()})
    arrays = dict(fixed_estimate=fixed[:, 0], fixed_stat_sigma=fixed[:, 1],
                  joint_estimate=joint[:, 0], joint_combined_sigma=joint[:, 1],
                  response_replica=replica_ids, response_expected_estimate=boot,
                  sampled_truth=np.asarray(sampled_truth), probability=probability)
    return row, arrays


def load_inputs(cfg, response_source):
    from scripts.run_tau_cij_unfolding import load_samples
    if not (response_source / 'manifest.json').exists():
        ready = json.loads((response_source / 'ready_samples.json').read_text())
        if ready.get('complete') is not True:
            raise ValueError('Expanded samples not ready')
        response_source = Path(ready['sample_directory'])
    panel, products, provenance = load_samples(Path(cfg['sample_source']))
    extra, extproducts, extprovenance = load_samples(response_source)
    if extprovenance.get('population') != 'filtered_full_validation_remainder':
        raise ValueError('Wrong expanded population')
    for arm in ('pretrain', 'dgpo'):
        a, b = provenance['arms'][arm], extprovenance['arms'][arm]
        if (Path(a['checkpoint']).resolve() != Path(b['checkpoint']).resolve()
                or a['global_step'] != b['global_step'] or a['epoch'] != b['epoch']):
            raise ValueError('Different models in response and test')
    for key in ('packing_spec', 'ddim_steps', 'process_label', 'weights'):
        if provenance.get(key) != extprovenance.get(key):
            raise ValueError(f'{key} differs')
    if np.intersect1d(panel['source_ids'], extra['source_ids']).size:
        raise ValueError('Response/test overlap')
    groups, group = np.unique(panel['kappas'], axis=0, return_inverse=True)
    egroups, egroup = np.unique(extra['kappas'], axis=0, return_inverse=True)
    if not np.array_equal(groups, egroups) or not np.isfinite(groups).all() or (groups == 0).any():
        raise ValueError('Invalid or different kappa groups')
    w, ew = (np.asarray(p['event_weight'], float) for p in (panel, extra))
    if any(not np.isfinite(v).all() or (v < 0).any() or v.sum() <= 0 for v in (w, ew)):
        raise ValueError('Invalid MC weights')
    rng = np.random.default_rng(cfg['seed'])
    old_response, test = [], []
    for g in range(len(groups)):
        ii = np.flatnonzero(group == g)
        rng.shuffle(ii)
        cut = int(len(ii) * cfg['response_fraction'])
        old_response.extend(ii[:cut]); test.extend(ii[cut:])
    return (panel, products, provenance, extra, extproducts, extprovenance,
            groups, group, egroup, w, ew, np.sort(old_response), np.sort(test))


def plot_report(report, output, run):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    for field in ('actual_cij', 'bias', 'coverage68'):
        fig = Figure(figsize=(13, 11), layout='constrained')
        FigureCanvasAgg(fig)
        for axis, ax in zip(AXES, fig.subplots(3, 3).ravel()):
            for arm, color in [('pretrain', 'tab:blue'), ('dgpo', 'tab:orange')]:
                rows = sorted((r for r in report['rows'] if r['component'] == axis
                               and r['arm'] == arm and r['scenario'] != 'nominal'),
                              key=lambda r: r['truth_exact'])
                x = [r['truth_exact'] for r in rows]
                if field == 'actual_cij':
                    y = [r['expected_unfolded'] for r in rows]
                    ax.errorbar(x, y, yerr=[r['expected_combined_sigma'] for r in rows],
                                fmt='o-', color=color, capsize=4, label=arm + ' stat+MC')
                    ax.errorbar(x, y, yerr=[r['expected_stat_sigma'] for r in rows],
                                fmt='none', color=color, elinewidth=3, alpha=.5)
                elif field == 'bias':
                    ax.plot(x, [r['stat_mean_residual'] for r in rows], 'o-', color=color, label=arm)
                else:
                    ax.plot(x, [r['stat_coverage68'] for r in rows], 'o--', color=color, label=arm + ' stat')
                    ax.plot(x, [r['joint_bootstrap_coverage68'] for r in rows], 's-', color=color,
                            label=arm + ' joint bootstrap')
            if field == 'actual_cij':
                ax.plot(report['config']['targets'], report['config']['targets'], 'k--', label='ideal')
            else:
                ax.axhline(0 if field == 'bias' else .6827, color='k', ls='--')
            ax.set(title=axis, xlabel='Injected truth Cij', ylabel={
                'actual_cij': 'Unfolded Cij (expected data)', 'bias': 'Mean unfolded − truth',
                'coverage68': 'Fraction covering truth'}[field])
            if field == 'coverage68': ax.set_ylim(-.03, 1.03)
        fig.axes[0].legend(fontsize=7)
        fig.suptitle(field + ' | fixed nominal response | pretrain vs DGPO')
        path = output / (field + '.png'); fig.savefig(path, dpi=150)
        if run:
            import wandb
            run.log({'bias_scan/plots/' + field: wandb.Image(str(path))})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('config', type=Path)
    p.add_argument('--response-source', type=Path, required=True)
    p.add_argument('--no-wandb', action='store_true')
    args = p.parse_args(); cfg = yaml.safe_load(args.config.read_text())
    B, R = int(cfg['repeats']), int(cfg['response_bootstraps'])
    if min(B, R) < 2 or not cfg['targets'] or not np.isfinite(cfg['targets']).all():
        raise ValueError('Need finite targets and at least two data and response replicas')
    import ROOT
    ROOT.gROOT.SetBatch(True)
    if not hasattr(ROOT, 'RooUnfoldSvd'):
        ROOT.gSystem.Load(cfg.get('roounfold_library') or 'libRooUnfold')
    if not hasattr(ROOT, 'RooUnfoldSvd'): raise RuntimeError('RooUnfold unavailable')
    (panel, products, provenance, extra, extproducts, extprovenance,
     groups, group, egroup, w, ew, old_response, test) = load_inputs(cfg, args.response_source)
    N = len(test)
    output = Path(cfg['output']) / ('large-response-bias-' + uuid.uuid4().hex[:10])
    output.mkdir(parents=True)
    np.savez_compressed(output / 'identities.npz', test_ids=panel['source_ids'][test],
                        binning_ids=panel['source_ids'][old_response], response_ids=extra['source_ids'])
    report = dict(complete=False, config=cfg, source=provenance, response_source=extprovenance,
                  test_events=N, response_events=len(ew), repeats=B, response_bootstraps=R,
                  rows=[], binning={}, scope=(
                      'Each Cij scanned separately by truth-only exponential-tilt event resampling. '
                      'Other moments may also change. Fixed nominal response, original response-only '
                      'binning, fixed SVD k. No truth-dependent retuning or bias subtraction. '
                      'Stat coverage conditions on the empirical test population and nominal response. '
                      'Joint coverage is approximate data+response bootstrap coverage: finite R response '
                      'replicas reused across B trials; not independent fresh MC ensembles. '
                      'Combined errors use quadrature, exclude detector/physics systematics and residual '
                      'bias. No acceptance correction or full physical spin-convention certification. '
                      'Finite test-panel resampling cannot establish unconditional physical bias.'))
    run = None
    try:
        if not args.no_wandb:
            import wandb
            settings = dict(cfg['wandb'])
            settings.pop('id', None); settings.pop('resume', None)
            settings['name'] = 'Does Cij track injected truth? | pretrain vs DGPO | large response | bias scan'
            run = wandb.init(**settings, config=dict(**cfg, response_source=str(args.response_source)), dir=str(output))
        for j, axis in enumerate(AXES):
            edges, info = response_edges({a: v[old_response, j] for a, v in products.items()},
                                        group[old_response], w[old_response], cfg['bins'],
                                        cfg.get('min_response_entries', 20))
            report['binning'][axis] = info
            nb = len(edges) - 1; k = min(int(cfg['svd_k']), nb)
            coefficients = ((9 / np.prod(groups, axis=1))[:, None] * ((edges[:-1] + edges[1:]) / 2)).ravel()
            z = 9 * products['truth'][test, j] / np.prod(panel['kappas'][test], axis=1)
            truthbins = group[test] * nb + indices(products['truth'][test, j], edges)
            scenarios = [('nominal', None, w[test] / w[test].sum(), 0.)]
            for target_index, target in enumerate(cfg['targets']):
                probability, tilt = target_probabilities(z, w[test], target)
                scenarios.append((f'target{target_index}', float(target), probability, tilt))
            for arm in ('pretrain', 'dgpo'):
                ti = indices(extproducts['truth'][:, j], edges)
                ri = indices(extproducts[arm][:, j], edges)
                banks = []
                # Identical event multipliers for both models/all components.
                rng = np.random.default_rng(cfg['seed'] + 2701)
                for replica in range(R + 1):
                    if replica % 5 == 0 or replica == R:
                        print(f'{axis} {arm}: building response {replica}/{R} '
                              '(0 = nominal)', flush=True)
                    weights = ew if replica == 0 else ew * rng.poisson(1, size=len(ew))
                    bank = []
                    for g in range(len(groups)):
                        take = egroup == g
                        bank.append(SVDResponse(ROOT, ti[take], ri[take], weights[take], nb))
                    banks.append(bank)
                reco = group[test] * nb + indices(products[arm][test, j], edges)

                def estimate(counts, replica):
                    values = np.empty(len(coefficients)); cov = np.zeros((len(coefficients), len(coefficients)))
                    for g, bank in enumerate(banks[replica]):
                        block = slice(g * nb, (g + 1) * nb)
                        values[block], cov[block, block] = bank.unfold(counts[block], k)
                    return moment(values, cov, coefficients)

                for s, (scenario, target, probability, tilt) in enumerate(scenarios):
                    row, arrays = evaluate_case(estimate, reco, coefficients, z, probability, N, B, R,
                                                cfg['seed'] + 1701 + 100 * j + 10000 * s)
                    binned = float(probability @ coefficients[truthbins])
                    row.update(component=axis, arm=arm, scenario=scenario, target=target, tilt=tilt,
                               bins=nb, k=k, truth_binned=binned, binning_shift=binned-row['truth_exact'],
                               expected_bias_binned=row['expected_unfolded']-binned)
                    report['rows'].append(row)
                    np.savez_compressed(output / f'{axis}_{arm}_{scenario}.npz', **arrays)
                    (output / 'bias_scan_report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
                    print(f'{axis} {arm} {scenario}: truth={row["truth_exact"]:+.3f} '
                          f'unfolded={row["expected_unfolded"]:+.3f} '
                          f'stat={row["expected_stat_sigma"]:.3f} MC={row["response_mc_sigma"]:.3f}', flush=True)
                    if run:
                        run.log({'bias_scan/' + key: value for key, value in row.items() if isinstance(value, (int, float))})
                        run.summary.update(dict(component=axis, arm=arm, scenario=scenario, completed_cases=len(report['rows'])))
                del banks
        plot_report(report, output, run)
        report['complete'] = True
        if run:
            keys = list(report['rows'][0])
            run.log({'bias_scan/results': wandb.Table(columns=keys, data=[[r[key] for key in keys] for r in report['rows']])})
            run.summary.update(dict(complete=True, repeats=B, response_bootstraps=R,
                                    test_events=N, response_events=len(ew)))
    finally:
        path = output / 'bias_scan_report.json'
        path.write_text(json.dumps(report, indent=2, allow_nan=False))
        print('REPORT:', path, flush=True)
        if run:
            run.save(str(path), base_path=str(output))
            run.finish(exit_code=0 if report['complete'] else 1)


if __name__ == '__main__':
    main()
