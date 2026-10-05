"""Exchange event masses and within-event proportions on the saved 16-GPU K64 panel."""
import argparse
import json
from pathlib import Path
import sys
import uuid

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.diagnose_tau_cij_components import load_source, verify_replay
from scripts.tau_weight_factor_swap import analyze, verify_endpoint, ARMS, AXES
from scripts.tau_tail_attribution import candidate_weights

PROTOCOL_KEYS = (
    'seed', 'workers', 'batch_size', 'epochs', 'lr', 'min_lr', 'hidden', 'dropout',
    'weight_decay', 'patience', 'min_delta', 'min_steps', 'head_kind', 'head_depth',
    'relative_dim', 'relative_preprocessing', 'packing_spec', 'condition_normalization',
    'backbone_manifest', 'train_source', 'test_source', 'train_events', 'test_events',
    'split_counts', 'condition_pt_edges', 'ratio_objective', 'mmd_coefficient',
    'ratio_bound', 'fresh_negatives', 'baseline_run', 'panel_run', 'condition_width',
    'condition_hidden', 'selector')


def read_settings(path):
    cfg = yaml.safe_load(Path(path).read_text())
    if (cfg['expected_events'], cfg['expected_candidates'], cfg['source_workers'], cfg['cap']) != (119002, 64, 16, 30):
        raise ValueError('Requires completed filtered 119002-event K64, 16-GPU, bound30 study')
    if cfg['bootstrap'] < 2000 or cfg['cpu_threads'] < 1:
        raise ValueError('Requires at least 2000 event bootstrap draws and positive CPU threads')
    name = cfg['logger']['name']
    if len(name) > 96 or not 3 <= len(name.split(' | ')) <= 5:
        raise ValueError('Invalid W&B display name')
    return cfg


def read_model(spec, ids, cfg, previous):
    path = Path(spec['directory'])
    if not (path/'COMPLETE').is_file():
        raise ValueError('Incomplete model result: '+str(path))
    if json.loads((path/'wandb.json').read_text())['id'] != spec['run']:
        raise ValueError('Model run ID mismatch')
    manifest = json.loads((path/'manifest.json').read_text())
    if (manifest['workers'], manifest['condition_width'], manifest['ratio_bound'], manifest['panel_run']) != (16, 256, 30, cfg['source_run']):
        raise ValueError('Model source protocol mismatch')
    if manifest['train_events'] != cfg['train_events'] or manifest['test_events'] != cfg['events']:
        raise ValueError('Model population differs from filtered dataset')
    if (manifest['backbone_manifest'].get('global_step') != 1110
            or manifest['train_manifest']['weights'] != 'raw_state_dict_only'
            or Path(manifest['panel_directory']).resolve() != Path(cfg['source']).resolve()):
        raise ValueError('Expected raw1110 and the exact pinned K64 source')
    with np.load(path/'fixed_panel_scores.npz', allow_pickle=False) as f:
        if not np.array_equal(ids, f['source_ids']):
            raise ValueError('A/B score identities/order differ from fixed panel')
        logits = f['logits'].astype(float)
    if (logits.shape != (cfg['expected_events'], 64) or not np.isfinite(logits).all()
            or logits.max() > np.log(30)+1e-6):
        raise ValueError('Scores are not complete finite bound30 log ratios')
    report = json.loads((path/'fixed_K64_report.json').read_text())
    if (report['events'], report['candidates'], report['ratio_bound']) != (cfg['expected_events'], 64, 30):
        raise ValueError('Incomplete fixed-panel report')
    replay = dict(truth=np.asarray(report['truth_C']).reshape(9).tolist(), arms=[])
    for key, name in (('unweighted', 'unweighted'), ('old_raw', 'raw'), ('old_cap30', 'cap30')):
        row = report['arms'][key]
        replay['arms'].append(dict(arm=name, cij=np.asarray(row['C']).reshape(9).tolist(),
                                  error=row['error'], event_ess=row['event_ess']))
    verify_replay(replay, previous)
    return manifest, logits, report


def verify_protocol(a, b):
    for key in PROTOCOL_KEYS:
        if key not in a or key not in b or a[key] != b[key]:
            raise ValueError('A/B training protocol differs: '+key)
    if a.get('explicit_input') or b.get('explicit_input', {}).get('arm') != 'geometry':
        raise ValueError('Expected A=context256, B=geometry-input intervention')


def preflight_endpoint(inputs, generated, scores, saved):
    """Catch source/score mismatches before the more expensive bootstrap."""
    w, logmean = candidate_weights(inputs['weight'], scores)
    mass = w.sum(1)
    truth = np.average(inputs['truth_cij'], weights=inputs['weight'], axis=0)
    c = np.einsum('nk,nkd->d', w, generated)
    row = dict(C=c.reshape(3, 3), error=np.linalg.norm(c-truth), event_ess=1/np.square(mass).sum(),
        candidate_ess=1/np.square(w).sum(), max_candidate_mass=w.max(), log_mean_ratio=logmean)
    verify_endpoint(row, truth.reshape(3, 3), saved)


def plots(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(13, 4), layout='constrained')
    axs[0].bar(ARMS, [report['arms'][a]['error'] for a in ARMS])
    axs[0].set(ylabel='Cij Frobenius error (lower better)', title='Same candidates; only weight factors exchanged')
    names = list(report['contrasts'])
    rows = list(report['contrasts'].values())
    axs[1].hlines(range(6), [r['simultaneous_ci95'][0] for r in rows],
                  [r['simultaneous_ci95'][1] for r in rows], color='tab:blue')
    axs[1].scatter([r['error_change'] for r in rows], range(6))
    axs[1].axvline(0, color='black', linestyle='--')
    axs[1].set(yticks=range(6), yticklabels=names, xlabel='Error change (negative better)',
               title='Paired bootstrap bands: primary2 / secondary family6')
    fig.savefig(output/'factor_swap.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 4), layout='constrained')
    for name in ARMS:
        ax.plot(AXES, report['arms'][name]['absolute_component_error'], marker='o', label=name)
    ax.set(ylabel='Absolute Cij error', title='All nine components; uncertainty in report / W&B table')
    ax.legend(ncol=5)
    fig.savefig(output/'components.png', dpi=160); plt.close(fig)


def publish(run, report, output):
    import wandb
    for name, row in report['arms'].items():
        run.summary.update({f'{name}/{k}': row[k] for k in
            ('error', 'candidate_ess', 'event_ess', 'max_candidate_mass', 'group_mass_tv', 'base_weighted_group_error')})
    rows, components = [], []
    for name, r in report['contrasts'].items():
        lo, hi = r['simultaneous_ci95']
        run.summary.update({name+'/error_change': r['error_change'], name+'/lo95': lo,
                            name+'/hi95': hi, name+'/status': r['status']})
        rows.append([name, r['error_change'], *r['pointwise_ci95'], lo, hi, r['family'], r['status']])
        for c in r['components']:
            components.append([name, c['component'], c['absolute_error_change'], *c['simultaneous_ci95'], c['status']])
    run.log({'Cij/factor_contrasts': wandb.Table(columns=['contrast', 'error_change', 'point_lo95', 'point_hi95',
        'simultaneous_lo95', 'simultaneous_hi95', 'family', 'status'], data=rows),
        'Cij/components': wandb.Table(columns=['contrast', 'component', 'absolute_error_change',
        'family54_lo95', 'family54_hi95', 'status'], data=components),
        'Cij/groups': wandb.Table(columns=['arm', 'category', 'pt_bin', 'events', 'base_mass', 'weighted_mass', 'error'],
        data=[[r[k] for k in ('arm', 'category', 'pt_bin', 'events', 'base_mass', 'weighted_mass', 'error')]
              for r in report['groups']])})
    for name in ('factor_swap.png', 'components.png'):
        run.log({'Cij/'+name[:-4]: wandb.Image(str(output/name))})
    for name in ('factor_swap_report.json', 'config.json', 'factor_swap.png', 'components.png'):
        run.save(str(output/name), base_path=str(output), policy='now')
    run.summary.update(dict(phase='complete', endpoints_verified=True, unique_root_cause_identified=False,
        generated_samples=0, classifier_fits=0, policy_updates=0, report_path=str(output/'factor_swap_report.json')))


def execute(cfg, output, run=None):
    manifest, inputs, data, previous, filtered = load_source(cfg)
    from evenet_dgpo.evenet.dataset.filtered_data import validate_filtered_dataset
    train_filtered = validate_filtered_dataset(cfg['train_events'])
    if train_filtered['rows'] != 416701:
        raise ValueError('Filtered training population differs')
    with np.load(Path(cfg['source'])/'inputs.npz', allow_pickle=False) as f:
        inputs['visible_pt_sum'] = f['visible_pt_sum']
    a, sa, ra = read_model(cfg['A'], inputs['source_ids'], cfg, previous)
    b, sb, rb = read_model(cfg['B'], inputs['source_ids'], cfg, previous)
    verify_protocol(a, b)
    preflight_endpoint(inputs, data['cij'], sa, ra)
    preflight_endpoint(inputs, data['cij'], sb, rb)
    settings = dict(cfg, condition_pt_edges=a['condition_pt_edges'])
    def progress(stage):
        print('[weight factor swap]', stage, flush=True)
        if run: run.log({'phase': stage})
    report, arrays = analyze(inputs, data['cij'], sa, sb, settings, progress)
    verify_endpoint(report['arms']['AA'], report['truth_C'], ra)
    verify_endpoint(report['arms']['BB'], report['truth_C'], rb)
    u = report['arms']['unweighted']
    verify_replay(dict(truth=np.asarray(report['truth_C']).reshape(9).tolist(), arms=[dict(
        arm='unweighted', cij=np.asarray(u['C']).reshape(9).tolist(), error=u['error'], event_ess=u['event_ess'])]), previous)
    report.update(settings=settings, source_manifest=manifest, model_manifests=dict(A=a, B=b),
        filtered_inputs=dict(train=train_filtered, validation=filtered), endpoints_verified=True,
        generated_samples=0, classifier_fits=0, policy_updates=0,
        convention='Inherited saved Cij and analyzing-power convention; not independently recertified here.')
    np.savez_compressed(output/'event_factors_and_bootstrap.npz', **arrays)
    (output/'factor_swap_report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    plots(report, output)
    if run: publish(run, report, output)
    (output/'COMPLETE').write_text('Read-only saved-panel diagnostic completed\n')
    print(json.dumps(report['interpretation'], indent=2), flush=True)
    print('REPORT:', output/'factor_swap_report.json', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--no-wandb', action='store_true')
    args = parser.parse_args(); cfg = read_settings(args.config)
    output = Path(cfg['output_root'])/('swap-'+uuid.uuid4().hex[:10])
    output.mkdir(parents=True)
    (output/'config.json').write_text(json.dumps(cfg, indent=2)+'\n')
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=cfg['cpu_threads']):
        if args.no_wandb:
            execute(cfg, output)
        else:
            import wandb
            with wandb.init(**cfg['logger'], config=cfg, dir=str(output), mode='online',
                tags=['Cij', 'factor-swap', 'saved-16-GPU-K64', 'CPU-analysis', 'no-training']) as run:
                (output/'wandb.json').write_text(json.dumps(dict(id=run.id, url=run.url))+'\n')
                run.summary.update(dict(phase='loading', generated_samples=0, classifier_fits=0, policy_updates=0))
                try: execute(cfg, output, run)
                except BaseException:
                    run.summary['phase'] = 'failed'
                    raise
                print('WANDB:', run.url, flush=True)


if __name__ == '__main__': main()
