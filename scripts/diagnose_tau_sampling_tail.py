"""User-launched CPU replay of saved K64 candidates; source files are never modified."""
from pathlib import Path
import argparse
import json
import sys
import uuid

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.tau_tail_attribution import AXES, analyze_prefix


def read_settings(path):
    cfg = yaml.safe_load(Path(path).read_text())
    if (cfg['prefixes'] != [32, 64] or cfg['expected_candidates'] != 64
            or cfg['expected_events'] != 119002 or cfg['bootstrap'] < 20
            or cfg['cpu_threads'] < 1 or cfg['top_count'] < 1
            or cfg['minimum_truth_bin_count'] < 1):
        raise ValueError('Requires full saved K64 population, K32/K64 comparisons and valid analysis settings')
    if (not cfg['ratio_caps'] or not all(np.isfinite(c) and c > 0 for c in cfg['ratio_caps'])
            or not cfg['temperatures'] or not all(np.isfinite(a) and 0 < a < 1 for a in cfg['temperatures'])
            or not cfg['remove_counts'] or not all(isinstance(n, int) and n > 0 for n in cfg['remove_counts'])):
        raise ValueError('Invalid sensitivity grid')
    if len(cfg['logger']['name']) > 96 or len(cfg['logger']['name'].split(' | ')) < 3:
        raise ValueError('Invalid W&B display name')
    return cfg


def load_source(cfg):
    source = Path(cfg['source'])
    if not (source/'COMPLETE').is_file():
        raise ValueError('Require completed sampling report')
    if json.loads((source/'wandb.json').read_text())['id'] != cfg['source_run']:
        raise ValueError('Source run differs')
    manifest = json.loads((source/'manifest.json').read_text())
    if (manifest['weights'] != 'raw_state_dict_only' or manifest['classifier_run'] != 'pzq0nl1i'
            or manifest['workers'] != 16 or manifest['classifier_fits'] != 0 or manifest['policy_updates'] != 0
            or manifest['conditions'] != cfg['expected_events'] or max(manifest['prefixes']) != cfg['expected_candidates']):
        raise ValueError('Source protocol differs from frozen raw1110 / FiLM K64 panel')
    with np.load(source/'inputs.npz', allow_pickle=False) as f:
        inputs = {k: f[k] for k in ('source_ids', 'weight', 'truth_cij', 'truth_tau', 'category',
                                  'visible_pt_sum', 'kappas', 'visible_a', 'visible_b')}
    with np.load(source/'samples_and_scores.npz', allow_pickle=False) as f:
        if not np.array_equal(f['source_ids'], inputs['source_ids']):
            raise ValueError('Candidate/input identities or ordering differ')
        data = {k: f[k] for k in ('logits', 'cij', 'tau', 'deltas')}
    n, k = cfg['expected_events'], cfg['expected_candidates']
    if len(inputs['source_ids']) != n or len(np.unique(inputs['source_ids'])) != n:
        raise ValueError('Incomplete or duplicate source IDs')
    for name, shape in {'weight': (n,), 'truth_cij': (n, 9), 'truth_tau': (n, 15), 'category': (n,),
                        'visible_pt_sum': (n,), 'kappas': (n, 2), 'visible_a': (n, 4), 'visible_b': (n, 4)}.items():
        if inputs[name].shape != shape or not np.isfinite(inputs[name]).all():
            raise ValueError('Invalid input '+name)
    for name, shape in {'logits': (n, k), 'cij': (n, k, 9), 'tau': (n, k, 15), 'deltas': (n, k, 2, 2)}.items():
        if data[name].shape != shape or not np.isfinite(data[name]).all():
            raise ValueError('Invalid candidate '+name)
    if (inputs['weight'] < 0).any() or inputs['weight'].sum() <= 0 or (inputs['kappas'] == 0).any():
        raise ValueError('Invalid weights/analyzing powers')
    prior = json.loads((source/'convergence_report.json').read_text())
    if prior['events'] != n or prior['candidates'] != k or not prior['extension']['old_prefixes_reproduced']:
        raise ValueError('Source convergence report differs')
    return manifest, inputs, data, prior


def verify_raw(result, prior):
    for name in ('unweighted', 'raw'):
        row = next(r for r in result['arms'] if r['arm'] == name)
        old = 'reweighted' if name == 'raw' else 'unweighted'
        if not np.allclose(row['cij'], prior[old], atol=1e-8, rtol=1e-8):
            raise ValueError('Original '+old+' Cij did not reproduce')
        if not np.isclose(row['error'], prior[old+'_error'], atol=1e-8, rtol=1e-8):
            raise ValueError('Original '+old+' error did not reproduce')
        if name == 'raw':
            for key in ('candidate_ess', 'event_ess', 'max_candidate_mass'):
                if not np.isclose(row[key], prior[key], atol=1e-8, rtol=1e-8):
                    raise ValueError('Original raw '+key+' did not reproduce')


def closure_metric(rows):
    # Three equal-priority tau histograms; average stratum TV weighted by base population.
    return float(sum(.5*abs(r['paired_delta'])*r['base_group_mass']
                     for r in rows if r['group'] != 'all')/3)


def log_prefix(run, k, result):
    import wandb
    for index, row in enumerate(result['arms']):
        metrics = {f'K{k}/{key}': value for key, value in row.items()
                   if isinstance(value, (int, float)) and not isinstance(value, bool)}
        metrics.update({f'K{k}/arm_index': index, f'K{k}/arm': row['arm'],
                        f'K{k}/conditional_tau_tv': closure_metric(result['closure'][row['arm']])})
        run.log(metrics)
        run.summary.update({f'K{k}/{row["arm"]}/{key}': value for key, value in row.items()
                            if isinstance(value, (int, float)) and not isinstance(value, bool)})
    records = result['attribution']['top_candidates']
    columns = ['source_id', 'draw', 'category', 'log_ratio', 'mass', 'influence_norm']
    run.log({f'K{k}/top_candidates': wandb.Table(columns=columns+list(AXES),
        data=[[r[c] for c in columns]+r['influence'] for r in records])})
    raw = result['closure']['raw']
    columns = ['group', 'family', 'bin', 'truth_count', 'generated_count', 'weighted_support_events',
               'weighted_bin_event_ess', 'sparse',
               'truth_probability', 'generated_probability', 'paired_delta', 'delta_lo95', 'delta_hi95',
               'base_group_mass', 'weighted_group_mass', 'truth_global_mass', 'generated_global_mass',
               'required_coarse_ratio', 'applied_coarse_ratio']
    run.log({f'K{k}/raw_coarse_closure': wandb.Table(columns=columns, data=[[r[c] for c in columns] for r in raw])})
    removals = result['attribution']['removal_sensitivity']
    columns = ['ranking', 'removed', 'removed_mass', 'error']
    run.log({f'K{k}/leave_out_sensitivity': wandb.Table(columns=columns,
             data=[[r[c] for c in columns] for r in removals if r['valid']])})


def plot_results(results, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(len(results), 3, figsize=(16, 4*len(results)), squeeze=False, layout='constrained')
    for ax, (k, result) in zip(axs, results.items()):
        rows = result['arms']
        x = np.arange(len(rows))
        vals = [r['error_minus_unweighted'] for r in rows]
        ci = np.array([r['error_minus_unweighted_ci95'] for r in rows])
        ax[0].vlines(x, ci[:, 0], ci[:, 1]); ax[0].scatter(x, vals)
        ax[0].axhline(0, color='black', linestyle='--')
        ax[0].set_title(f'K{k}: Cij error minus unweighted / paired 95% CI')
        ax[1].plot(x, [r['event_ess'] for r in rows], 'o-'); ax[1].set_yscale('log')
        ax[1].set_title('Event ESS (not NK independent events)')
        ax[2].plot(x, [100*r['max_candidate_mass'] for r in rows], 'o-')
        ax[2].set_title('Largest candidate mass (%)')
        for a in ax:
            a.set_xticks(x, [r['arm'] for r in rows], rotation=60, ha='right')
    fig.suptitle('Fixed models / saved panel: transformed ratios are sensitivity tests, NOT an unbiased correction')
    fig.savefig(path, dpi=150); plt.close(fig)


def execute(cfg, output, run=None):
    manifest, inputs, data, prior = load_source(cfg)
    report = dict(source_run=cfg['source_run'], source_directory=str(Path(cfg['source']).resolve()),
        source_manifest=manifest, settings=cfg, classifier_fits=0, policy_updates=0, generated_samples=0,
        candidates=cfg['expected_candidates'], events=cfg['expected_events'], prefixes={},
        convention='Existing matched Cij features retained exactly; raw ratio globally normalized with base_weight/K.',
        limitations=['Exploratory on previously inspected validation conditions, not a new independent test.',
            'Cap/tempering and candidate removal change the target. No best arm is selected or deployed.',
            'Paired event bootstrap keeps truth/candidates together, fixed models and observed support only.',
            'Histogram intervals are pointwise event-cluster delta approximations; sparse bins flagged.',
            'No unseen-tail, classifier-fit or multiple-testing uncertainty is included.',
            'Coarse tau-direction bins can expose mismatches but cannot certify full conditional ratios or spin closure.'],
        bins=dict(joint_costheta_edges=[-1, -.5, 0, .5, 1], joint_phi_edges=np.linspace(-np.pi, np.pi, 5).tolist(),
                  opening_cos_edges=np.linspace(-1, 1, 9).tolist(), condition_pt_edges=manifest['condition_pt_edges'],
                  condition='Decay category x fit-only visible-pT quartiles, same saved edges; also all events'))
    def progress(k, name):
        print(f'[tail attribution] K={k} arm={name}: moments and coarse closure complete', flush=True)
        if run:
            run.log({'progress/K': k, 'progress/arm': name})
    for k in cfg['prefixes']:
        result = analyze_prefix(inputs, data, k, cfg, manifest['condition_pt_edges'], progress)
        verify_raw(result, prior['prefixes'][str(k)])
        result['original_raw_and_unweighted_verified'] = True
        report['prefixes'][str(k)] = result
        # Preserve a complete K32 report even if the later K64 analysis is interrupted.
        partial = output/f'K{k}_report.json'
        partial.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        if run:
            log_prefix(run, k, result)
            run.save(str(partial), base_path=str(output), policy='now')
    path = output/'tail_attribution_report.json'
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    plot_results(report['prefixes'], output/'sensitivity.png')
    if run:
        import wandb
        run.log({'tail/sensitivity': wandb.Image(str(output/'sensitivity.png'))})
        run.save(str(path), base_path=str(output), policy='now')
        run.save(str(output/'sensitivity.png'), base_path=str(output), policy='now')
        run.summary.update(dict(phase='complete', original_endpoints_verified=True,
                                classifier_fits=0, policy_updates=0, generated_samples=0, report_path=str(path)))
    (output/'COMPLETE').write_text('Saved-panel analysis complete; source unchanged\n')
    print('REPORT:', path, flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('config', type=Path)
    p.add_argument('--no-wandb', action='store_true')
    args = p.parse_args()
    cfg = read_settings(args.config)
    output = Path(cfg['output_root'])/('tail-'+uuid.uuid4().hex[:10])
    output.mkdir(parents=True)
    (output/'config.json').write_text(json.dumps(cfg, indent=2)+'\n')
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=cfg['cpu_threads']):
        if args.no_wandb:
            execute(cfg, output)
        else:
            import wandb
            logger = cfg['logger']
            with wandb.init(entity=logger['entity'], project=logger['project'], name=logger['name'],
                            group=logger['group'], config=cfg, dir=str(output), mode='online',
                            tags=['saved-panel', 'CPU-analysis', 'raw-1110', 'frozen-FiLM', 'no-training']) as run:
                (output/'wandb.json').write_text(json.dumps(dict(id=run.id, url=run.url))+'\n')
                run.summary.update(dict(phase='loading_saved_panel', classifier_fits=0, policy_updates=0, generated_samples=0))
                for k in cfg['prefixes']:
                    run.define_metric(f'K{k}/arm_index')
                    run.define_metric(f'K{k}/*', step_metric=f'K{k}/arm_index')
                try:
                    execute(cfg, output, run)
                except BaseException:
                    run.summary['phase'] = 'failed'
                    raise
                print('WANDB:', run.url, flush=True)


if __name__ == '__main__':
    main()
