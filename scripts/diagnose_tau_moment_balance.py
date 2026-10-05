"""Physics-targeted, nine-coefficient moment calibration; no classifier refit.

One command reuses the saved16-GPU K64 samples and fixed context256 scores.
The new optimizer is a small CPU calibration, not DGPO or a GPU training job.
"""
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
from scripts.diagnose_tau_weight_factor_swap import read_settings as source_settings, read_model, preflight_endpoint
from scripts.tau_moment_balance import analyze, AXES


def read_settings(path):
    overlay = yaml.safe_load(Path(path).read_text())
    cfg = source_settings(ROOT / overlay['source_config'])
    cfg.pop('B')
    cfg.update(overlay)
    if cfg['A']['run'] != 'zrv2yfgt' or cfg['cap'] != 30 or cfg['source_workers'] != 16:
        raise ValueError('Requires the fixed context256 bound30 and16-GPU source')
    if (cfg['split_fractions'] != [.4, .2, .4] or cfg['minimum_split_events'] < 10000
            or cfg['moment_strengths'] != [.1, 1., 10.] or cfg['max_correction_factor'] != 2.
            or cfg['mass_strength'] != 0. or cfg['coefficient_ridge'] <= 0
            or cfg['coefficient_limit'] <= 0 or cfg['max_iterations'] < 1
            or cfg['bootstrap'] < 2000 or cfg['cpu_threads'] != 16):
        raise ValueError('Changed predeclared moment-calibration protocol')
    guards = cfg['guardrails']
    if not 0 < guards['ess_retention'] <= 1 or any(guards[k] < 0 for k in
            ('mass_tv_increase', 'diagonal_error_increase', 'total_error_increase')):
        raise ValueError('Invalid calibration guardrails')
    name = cfg['logger']['name']
    if len(name) > 96 or not 3 <= len(name.split(' | ')) <= 5:
        raise ValueError('Invalid W&B display name')
    return cfg


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def plots(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    test = report['test']
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    for name, row in test['arms'].items():
        axs[0].plot(AXES, row['absolute_component_error'], marker='o', label=name)
    axs[0].set(title='Held-out events: all nine Cij entries', ylabel='Absolute error')
    axs[0].legend()
    rows = [test['comparisons'][k] for k in ('offdiagonal', 'diagonal', 'total')]
    axs[1].hlines(range(3), [r['ci95'][0] for r in rows], [r['ci95'][1] for r in rows])
    axs[1].scatter([r['change'] for r in rows], range(3))
    axs[1].axvline(0, color='grey', ls='--')
    axs[1].set(yticks=range(3), yticklabels=['Cross terms (primary)', 'Diagonal', 'Full Cij'],
        xlabel='Calibrated minus baseline error (negative better)', title='Paired event bootstrap: 95% intervals')
    fig.savefig(output/'heldout_moments.png', dpi=160); plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    for name, model in report['models'].items():
        axs[0].plot([r['iteration'] for r in model['history']], [r['loss'] for r in model['history']], label=name)
    axs[0].set(xlabel='Calibration optimizer iteration', ylabel='Objective', title='Calibration split only')
    axs[0].legend()
    names = list(report['validation'])
    for key in ('offdiagonal_error', 'diagonal_error', 'error'):
        axs[1].plot(names, [report['validation'][n][key] for n in names], marker='o', label=key)
    axs[1].set(title='Selection split (NOT final test)', ylabel='Cij error')
    axs[1].tick_params(axis='x', rotation=20); axs[1].legend()
    fig.savefig(output/'calibration_selection.png', dpi=160); plt.close(fig)


def publish(run, report, output):
    import wandb
    test = report['test']
    run.summary.update({f'split/{k}/events': v for k, v in report['split_counts'].items()})
    run.summary.update({f'test/correction/{k}': v for k, v in test['correction'].items()})
    run.summary.update({f'test/guardrails/{k}': v for k, v in test['observed_guardrails'].items()})
    for split in ('calibration', 'validation'):
        for arm, row in report[split].items():
            run.summary.update({f'{split}/{arm}/{key}': val for key, val in row.items() if isinstance(val, (int, float))})
    for arm, row in test['arms'].items():
        run.summary.update({f'test/{arm}/{key}': val for key, val in row.items() if isinstance(val, (int, float))})
    for name in ('offdiagonal', 'diagonal', 'total'):
        r = test['comparisons'][name]
        run.summary.update({f'test/{name}/change': r['change'], f'test/{name}/lo95': r['ci95'][0],
            f'test/{name}/hi95': r['ci95'][1], f'test/{name}/status': r['status']})
    run.log({'test/components': wandb.Table(columns=['component', 'absolute_error_change', 'point_lo95',
        'point_hi95', 'family9_lo95', 'family9_hi95'], data=[
        [r['component'], r['change'], *r['pointwise_ci95'], *r['simultaneous_ci95']]
        for r in test['comparisons']['components']]),
        'test/condition_groups': wandb.Table(columns=['arm', 'category', 'pt_bin', 'events', 'base_mass', 'weighted_mass', 'error'],
            data=[[arm, *[r[k] for k in ('category', 'pt_bin', 'events', 'base_mass', 'weighted_mass', 'error')]]
                for arm, row in test['arms'].items() for r in row['groups']]),
        'selection/candidates': wandb.Table(columns=['arm', 'converged', 'candidate_ess', 'event_ess',
            'condition_mass', 'diagonal', 'total', 'offdiagonal_improves'], data=[
            [name, *[row[k] for k in ('converged', 'candidate_ess', 'event_ess', 'condition_mass',
                'diagonal', 'total', 'offdiagonal_improves')]] for name, row in report['selection']['candidates'].items()])})
    for name in ('heldout_moments.png', 'calibration_selection.png'):
        run.log({'plots/'+name[:-4]: wandb.Image(str(output/name))})
    for name in ('moment_balance_report.json', 'calibration_model.json', 'config.json',
                 'heldout_moments.png', 'calibration_selection.png'):
        run.save(str(output/name), base_path=str(output), policy='now')
    run.summary.update(dict(phase='complete', selected=report['selection']['selected'],
        qualified_improvement=test['qualified_improvement'], source_endpoints_verified=True,
        classifier_fits=0, calibration_fits=report['calibration_fits'], generated_samples=0, policy_updates=0,
        pristine_test=False, report_path=str(output/'moment_balance_report.json')))


def execute(cfg, output, run=None):
    source, inputs, data, previous, filtered = load_source(cfg)
    from evenet_dgpo.evenet.dataset.filtered_data import validate_filtered_dataset
    train = validate_filtered_dataset(cfg['train_events'])
    if train['rows'] != 416701:
        raise ValueError('Filtered training population differs')
    with np.load(Path(cfg['source'])/'inputs.npz', allow_pickle=False) as f:
        inputs['visible_pt_sum'] = f['visible_pt_sum']
    manifest, scores, saved = read_model(cfg['A'], inputs['source_ids'], cfg, previous)
    if manifest.get('explicit_input'):
        raise ValueError('Baseline must be unchanged context256, without explicit-product features')
    preflight_endpoint(inputs, data['cij'], scores, saved)
    settings = dict(cfg, condition_pt_edges=manifest['condition_pt_edges'])

    def progress(stage, values):
        print('[moment balance]', stage, json.dumps(values, allow_nan=False), flush=True)
        if run:
            prefix = f'fit/moment_{values["strength"]:g}' if stage == 'calibration_fit' else stage
            run.log(dict(phase=stage, **{prefix+'/'+k: v for k, v in values.items()}))

    def seal(frozen, split):
        write_json(output/'calibration_model.json', frozen)
        np.savez_compressed(output/'split_ids.npz', source_ids=inputs['source_ids'], event_split=split)
        if run:
            run.save(str(output/'calibration_model.json'), base_path=str(output), policy='now')
            run.summary.update(dict(phase='selection_locked', selected=frozen['selection']['selected'], selection_uses_test=False))

    report, arrays = analyze(inputs, data['cij'], scores, settings, progress, seal)
    report.update(settings=settings, source_manifest=source, classifier_manifest=manifest,
        filtered_inputs=dict(train=train, validation=filtered), source_endpoints_verified=True,
        convention='Inherited saved Cij/analyzing-power convention; not independently recertified.',
        literature='https://www.mit.edu/~jhainm/Paper/eb.pdf',
        method='Soft regularized moment calibration with bounded nonlinear tilt, not exact convex entropy balancing.')
    np.savez_compressed(output/'heldout_event_bootstrap.npz', **arrays)
    write_json(output/'moment_balance_report.json', report)
    plots(report, output)
    if run:
        publish(run, report, output)
    (output/'COMPLETE').write_text('Split-held-out moment calibration completed; sources unchanged\n')
    print(json.dumps(dict(selected=report['selection']['selected'],
        qualified_improvement=report['test']['qualified_improvement'],
        primary=report['test']['comparisons']['offdiagonal']), indent=2), flush=True)
    print('REPORT:', output/'moment_balance_report.json', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--no-wandb', action='store_true')
    args = parser.parse_args()
    cfg = read_settings(args.config)
    output = Path(cfg['output_root'])/('balance-'+uuid.uuid4().hex[:10])
    output.mkdir(parents=True)
    write_json(output/'config.json', cfg)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=cfg['cpu_threads']):
        if args.no_wandb:
            execute(cfg, output)
        else:
            import wandb
            with wandb.init(**cfg['logger'], config=cfg, dir=str(output), mode='online',
                tags=['Cij', 'moment-calibration', 'physics-targeted', 'saved-16-GPU-K64', 'no-policy-update', 'exploratory-split']) as run:
                for strength in cfg['moment_strengths']:
                    prefix = f'fit/moment_{strength:g}'
                    run.define_metric(prefix+'/iteration')
                    run.define_metric(prefix+'/*', step_metric=prefix+'/iteration')
                write_json(output/'wandb.json', dict(id=run.id, url=run.url))
                run.summary.update(dict(phase='loading', classifier_fits=0, policy_updates=0,
                    generated_samples=0, pristine_test=False))
                try:
                    execute(cfg, output, run)
                except BaseException:
                    run.summary['phase'] = 'failed'
                    raise
                print('WANDB:', run.url, flush=True)


if __name__ == '__main__':
    main()
