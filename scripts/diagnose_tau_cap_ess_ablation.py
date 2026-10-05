"""Post-hoc caps and ESS-matched controls on completed zrv2yfgt K64 scores."""
import argparse
import json
from pathlib import Path
import sys
import uuid

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.diagnose_tau_cij_components import load_source
from scripts.diagnose_tau_weight_factor_swap import read_settings as read_source_settings, read_model, preflight_endpoint
from scripts.tau_weight_factor_swap import verify_endpoint
from scripts.tau_cap_ess_ablation import analyze, AXES


def read_settings(path):
    overlay = yaml.safe_load(Path(path).read_text())
    cfg = read_source_settings(ROOT/overlay['source_config'])
    cfg.pop('B')
    cfg.update(overlay)
    if cfg['caps'] != [30, 20, 10, 5] or cfg['A']['run'] != 'zrv2yfgt':
        raise ValueError('Requires fixed context256 scores and cap30/20/10/5')
    if cfg['bootstrap'] < 2000 or cfg['cpu_threads'] != 16:
        raise ValueError('Requires 2000+ event bootstrap draws and16 CPU threads')
    name = cfg['logger']['name']
    if len(name) > 96 or not 3 <= len(name.split(' | ')) <= 5:
        raise ValueError('Invalid W&B display name')
    return cfg


def plots(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(13, 4), layout='constrained')
    for family, marker in [('cap', 'o'), ('mix', 's')]:
        rows = [r for a, r in report['arms'].items() if a.startswith(family)]
        axs[0].plot([r['candidate_ess_fraction'] for r in rows], [r['error'] for r in rows],
                    marker=marker, label=family)
        for r in rows:
            axs[0].annotate(str(r.get('cap', r.get('matched_cap'))),
                            (r['candidate_ess_fraction'], r['error']))
    axs[0].axhline(report['arms']['unweighted']['error'], color='grey', ls='--', label='unweighted')
    axs[0].set(xlabel='Candidate ESS fraction', ylabel='Cij Frobenius error', title='Same ESS, different weights')
    axs[0].legend()
    names = list(report['contrasts'])
    rows = list(report['contrasts'].values())
    axs[1].hlines(range(len(rows)), [r['simultaneous_ci95'][0] for r in rows],
                  [r['simultaneous_ci95'][1] for r in rows])
    axs[1].scatter([r['error_change'] for r in rows], range(len(rows)))
    axs[1].axvline(0, color='grey', ls='--')
    axs[1].set(yticks=range(len(rows)), yticklabels=names,
               xlabel='Error change (negative better)', title='Paired simultaneous95% bands')
    fig.savefig(output/'cap_ess.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 4), layout='constrained')
    for name, r in report['arms'].items():
        ax.plot(AXES, r['absolute_component_error'], marker='o',
                ls='--' if name.startswith('mix') else '-', label=name)
    ax.set(ylabel='Absolute Cij error', title='All nine entries: do not hide diagonal/off-diagonal tradeoffs')
    ax.legend(ncol=4)
    fig.savefig(output/'components.png', dpi=160); plt.close(fig)


def publish(run, report, output):
    import wandb
    for name, row in report['arms'].items():
        run.summary.update({f'{name}/{key}': value for key, value in row.items()
                            if isinstance(value, (int, float, str))})
    for name, row in report['contrasts'].items():
        run.summary.update({name+'/error_change': row['error_change'], name+'/status': row['status'],
            name+'/lo95': row['simultaneous_ci95'][0], name+'/hi95': row['simultaneous_ci95'][1]})
    run.log({'Cij/components': wandb.Table(columns=['contrast', 'component', 'absolute_error_change',
        'simultaneous_lo95', 'simultaneous_hi95', 'status'], data=[
        [name, c['component'], c['absolute_error_change'], *c['simultaneous_ci95'], c['status']]
        for name, r in report['contrasts'].items() for c in r['components']]),
        'Cij/groups': wandb.Table(columns=['arm', 'grouping', 'category', 'pt_bin', 'events',
        'base_mass', 'weighted_mass', 'error'], data=[
        [r[key] for key in ('arm', 'grouping', 'category', 'pt_bin', 'events', 'base_mass', 'weighted_mass', 'error')]
        for r in report['groups']])})
    for name in ('cap_ess.png', 'components.png'):
        run.log({'Cij/'+name[:-4]: wandb.Image(str(output/name))})
    for name in ('cap_ess_report.json', 'config.json', 'cap_ess.png', 'components.png'):
        run.save(str(output/name), base_path=str(output), policy='now')
    run.summary.update(dict(phase='complete', source_endpoints_verified=True, classifier_fits=0,
        generated_samples=0, policy_updates=0, best_cap_selected=False,
        report_path=str(output/'cap_ess_report.json')))


def execute(cfg, output, run=None):
    manifest, inputs, data, previous, filtered = load_source(cfg)
    from evenet_dgpo.evenet.dataset.filtered_data import validate_filtered_dataset
    train = validate_filtered_dataset(cfg['train_events'])
    if train['rows'] != 416701: raise ValueError('Filtered training population differs')
    with np.load(Path(cfg['source'])/'inputs.npz', allow_pickle=False) as f:
        inputs['visible_pt_sum'] = f['visible_pt_sum']
    model, logits, saved = read_model(cfg['A'], inputs['source_ids'], cfg, previous)
    preflight_endpoint(inputs, data['cij'], logits, saved)
    settings = dict(cfg, condition_pt_edges=model['condition_pt_edges'])
    def progress(stage):
        print('[cap/ESS]', stage, flush=True)
        if run: run.log({'phase': stage})
    report, arrays = analyze(inputs, data['cij'], logits, settings, progress)
    verify_endpoint(report['arms']['cap30'], report['truth_C'], saved)
    report.update(settings=settings, source_manifest=manifest, model_manifest=model,
        filtered_inputs=dict(train=train, validation=filtered), source_endpoints_verified=True,
        classifier_fits=0, generated_samples=0, policy_updates=0,
        convention='Inherited saved Cij/analyzing-power convention; not independently recertified.')
    np.savez_compressed(output/'event_weights_and_bootstrap.npz', **arrays)
    (output/'cap_ess_report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    plots(report, output)
    if run: publish(run, report, output)
    (output/'COMPLETE').write_text('Fixed-score cap/ESS ablation completed; original files unchanged\n')
    print(json.dumps({name: r['status'] for name, r in report['contrasts'].items()}, indent=2), flush=True)
    print('REPORT:', output/'cap_ess_report.json', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--no-wandb', action='store_true')
    args = parser.parse_args(); cfg = read_settings(args.config)
    output = Path(cfg['output_root'])/('cap-ess-'+uuid.uuid4().hex[:10]); output.mkdir(parents=True)
    (output/'config.json').write_text(json.dumps(cfg, indent=2)+'\n')
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=cfg['cpu_threads']):
        if args.no_wandb: execute(cfg, output)
        else:
            import wandb
            with wandb.init(**cfg['logger'], config=cfg, dir=str(output), mode='online',
                    tags=['Cij', 'cap-sweep', 'ESS-matched', 'saved-16-GPU-K64', 'no-training']) as run:
                (output/'wandb.json').write_text(json.dumps(dict(id=run.id, url=run.url))+'\n')
                run.summary.update(dict(phase='loading', classifier_fits=0, generated_samples=0, policy_updates=0))
                try: execute(cfg, output, run)
                except BaseException:
                    run.summary['phase'] = 'failed'
                    raise
                print('WANDB:', run.url, flush=True)


if __name__ == '__main__': main()
