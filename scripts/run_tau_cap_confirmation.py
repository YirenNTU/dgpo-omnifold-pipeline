"""Fresh K64 sampling with a pre-fixed cap30; only user-launched, on16 GPUs."""
import argparse
import json
import os
from pathlib import Path
import sys
import uuid

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_tau_sampling_convergence import prepare, worker, merge
from scripts.tau_tail_attribution import candidate_weights, influence_norm, paired_bootstrap, AXES


def read_settings(path):
    cfg = yaml.safe_load(Path(path).read_text())
    if (cfg['workers'] != 16 or cfg['batch_size'] != 1024 or cfg['feature_batch_size'] != 256
            or cfg['prefixes'] != [64] or cfg['cap'] != 30 or cfg['expected_events'] != 119002
            or cfg['bootstrap'] < 20 or cfg.get('extend_from') or cfg.get('inherited_candidates', 0)):
        raise ValueError('Requires fresh K64, fixed cap30, full validation and16 GPUs/batch1024')
    if len(cfg['logger']['name']) > 96 or len(cfg['logger']['name'].split(' | ')) < 3:
        raise ValueError('Invalid W&B display name')
    return cfg


def stream_seeds(cfg):
    return {cfg['seed']+1000003*j+rank for j in range(max(cfg['prefixes'])) for rank in range(cfg['workers'])}


def validate_replication(cfg, inputs):
    """Freeze model, preprocessing, population and sampling; only RNG streams differ."""
    source = Path(cfg['source'])
    if not (source/'COMPLETE').is_file():
        raise ValueError('Source panel incomplete')
    if json.loads((source/'wandb.json').read_text())['id'] != cfg['source_run']:
        raise ValueError('Wrong source run')
    old = json.loads((source/'manifest.json').read_text())
    for key in ('classifier_run', 'classifier_checkpoint', 'generator_checkpoint', 'events', 'conditions',
                'runtime', 'workers', 'batch_size', 'feature_batch_size', 'ddim_steps', 'packing_spec',
                'weights', 'ratio_transform', 'historical_cij_errors', 'condition_pt_edges'):
        if key not in old or key not in cfg or cfg[key] != old[key]:
            raise ValueError('Replication protocol changed: '+key)
    if max(old['prefixes']) != 64 or stream_seeds(cfg)&stream_seeds(old):
        raise ValueError('Fresh RNG streams overlap the original K64 panel')
    with np.load(source/'inputs.npz', allow_pickle=False) as f:
        if set(f.files) != set(inputs):
            raise ValueError('Input schema differs')
        for name in f.files:
            if not np.array_equal(inputs[name], f[name]):
                raise ValueError('Replication conditions/truth changed: '+name)
    return dict(cfg, inherited_candidates=0, source_seed=old['seed'], seed_overlap_count=0,
        evaluation_status='Fresh Monte Carlo draws on previously inspected conditions; NOT independent-event generalization',
        primary_endpoint='K64 capped Cij norm error minus unweighted, paired event-bootstrap95% interval',
        criterion='Point difference <0 and upper paired95% bound <0; no cap selection or model changes')


def analyze(inputs, data, cfg):
    base, g, s = inputs['weight'], data['cij'], data['logits']
    target = np.average(inputs['truth_cij'], weights=base, axis=0)
    rows, nums, dens = [], [], []
    for name, logit in [('unweighted', np.zeros_like(s)), ('raw', s), ('cap30', np.minimum(s, np.log(cfg['cap'])))]:
        w, log_mean = candidate_weights(base, logit)
        num, den = np.einsum('nk,nkd->nd', w, g), w.sum(1)
        matrix = num.sum(0)
        influence = influence_norm(w, g, matrix)
        flat = int(np.argmax(influence)); i, j = divmod(flat, s.shape[1])
        mass = float(w[i, j])
        deleted = None if mass >= 1 else (matrix-mass*g[i, j])/(1-mass)
        rows.append(dict(arm=name, cij=matrix.tolist(), error=float(np.linalg.norm(matrix-target)),
            event_ess=float(1/np.square(den).sum()), candidate_ess=float(1/np.square(w).sum()),
            max_candidate_mass=float(w.max()), max_event_mass=float(den.max()), log_mean_ratio=log_mean,
            most_influential=dict(source_id=str(inputs['source_ids'][i]), draw=j+1,
                logit=float(s[i, j]), mass=mass, influence_norm=float(influence[i, j]),
                leave_one_error=None if deleted is None else float(np.linalg.norm(deleted-target))),
            sensitivity_only=name=='cap30'))
        nums.append(num); dens.append(den)
    estimates, targets = paired_bootstrap(inputs, nums, dens, cfg['bootstrap'], cfg['bootstrap_seed'])
    residual = estimates-targets[:, None]
    errors = np.linalg.norm(residual, axis=-1)
    for i, row in enumerate(rows):
        row['error_minus_unweighted'] = row['error']-rows[0]['error']
        row['error_minus_unweighted_ci95'] = np.quantile(errors[:, i]-errors[:, 0], [.025, .975]).tolist()
        row['matrix_ci95'] = np.quantile(estimates[:, i], [.025, .975], axis=0).tolist()
        row['residual'] = (np.array(row['cij'])-target).tolist()
        row['residual_ci95'] = np.quantile(residual[:, i], [.025, .975], axis=0).tolist()
    cap = rows[2]
    return dict(truth=target.tolist(), arms=rows,
        sampling_confirmation_passed=bool(cap['error_minus_unweighted']<0 and cap['error_minus_unweighted_ci95'][1]<0),
        independent_event_generalization_tested=False, cap_fixed_before_sampling=cfg['cap'],
        interpretation='Conditional Monte Carlo replication, fixed models and previously inspected conditions. '
        'Paired event-bootstrap intervals are pointwise, do not include unseen tails/model-fit uncertainty; '
        'passing is not conditional-distribution closure. Capping changes the ratio target.')


def finish(cfg, run):
    output = Path(cfg['output'])
    with np.load(output/'inputs.npz', allow_pickle=False) as f:
        inputs = {key: f[key] for key in ('source_ids', 'weight', 'truth_cij')}
    data, parity = merge(cfg, inputs['source_ids'])
    np.savez_compressed(output/'samples_and_scores.npz', source_ids=inputs['source_ids'], **data)
    report = analyze(inputs, data, cfg)
    report.update(settings=cfg, events=cfg['conditions'], candidates=max(cfg['prefixes']),
                  classifier_fits=0, policy_updates=0, historical_score_replay_max_error=parity)
    path = output/'cap_confirmation_report.json'
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    import wandb
    table = []
    for index, row in enumerate(report['arms']):
        metrics = {k: v for k, v in row.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        run.log(dict(arm_index=index, arm=row['arm'], **metrics))
        run.summary.update({row['arm']+'/'+key: value for key, value in metrics.items()})
        run.summary[row['arm']+'/error_minus_unweighted_ci95'] = row['error_minus_unweighted_ci95']
        for j, axis in enumerate(AXES):
            table.append([row['arm'], axis, report['truth'][j], row['cij'][j], row['residual'][j],
                          row['residual_ci95'][0][j], row['residual_ci95'][1][j]])
    run.log({'Cij/components': wandb.Table(columns=['arm','component','truth','estimate','residual','lo95','hi95'],data=table)})
    plot(report, output/'Cij_confirmation.png')
    run.log({'Cij/confirmation': wandb.Image(str(output/'Cij_confirmation.png'))})
    for file in (path, output/'Cij_confirmation.png'):
        run.save(str(file), base_path=str(output), policy='now')
    run.summary.update(dict(phase='complete', sampling_confirmation_passed=report['sampling_confirmation_passed'],
        independent_event_generalization_tested=False, historical_score_replay_max_error=parity,
        samples=cfg['conditions']*max(cfg['prefixes']), reused_samples=0, classifier_fits=0, policy_updates=0,
        report_path=str(path)))
    (output/'COMPLETE').write_text('Fixed-cap sampling confirmation complete; not an independent-event test\n')
    print('REPORT:', path, 'WANDB:', run.url, flush=True)


def plot(report, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    for i, row in enumerate(report['arms']):
        axs[0].scatter(i, row['error_minus_unweighted'])
        axs[0].vlines(i, *row['error_minus_unweighted_ci95'])
        axs[1].plot(np.arange(9), row['residual'], 'o-', label=row['arm'])
    axs[0].set(xticks=range(3), xticklabels=[r['arm'] for r in report['arms']], ylabel='Cij error minus unweighted',
               title='Paired95% interval; fixed cap30')
    axs[1].set(xticks=range(9), xticklabels=AXES, ylabel='Estimated Cij minus truth', title='All nine components')
    for ax in axs: ax.axhline(0, color='black', linestyle='--')
    axs[1].legend(); fig.suptitle('Fresh K64 draws / same conditions / frozen models')
    fig.savefig(path, dpi=160); plt.close(fig)


def sample(cfg, run, address):
    import ray
    ray.init(address=address, runtime_env={'env_vars': {'PYTHONPATH': os.pathsep.join(
        (str(ROOT), str(ROOT/'scripts'), str(ROOT/'evenet_dgpo'), os.environ.get('PYTHONPATH','')))}})
    pending = []
    try:
        if ray.cluster_resources().get('GPU', 0) < cfg['workers']:
            raise ValueError('Requires16 GPUs on existing Ray cluster')
        remote = ray.remote(num_gpus=1, num_cpus=1, max_calls=1, max_retries=0)(worker)
        pending = [remote.remote(cfg, rank) for rank in range(cfg['workers'])]
        while pending:
            done, pending = ray.wait(pending, num_returns=1, timeout=10)
            ray.get(done)
            progress = [json.loads(p.read_text()) for p in Path(cfg['output']).glob('progress-*.json')]
            run.log({'sampling/samples_complete': sum(r['samples'] for r in progress),
                     'sampling/finished_workers': cfg['workers']-len(pending)})
    finally:
        for ref in pending: ray.cancel(ref, force=True)
        ray.shutdown()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('config', type=Path)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--analyze-only', type=Path, help='Reuse this confirmation run after sampling; no resampling')
    p.add_argument('--ray-address', default=os.environ.get('RAY_ADDRESS') or 'auto')
    args = p.parse_args()
    requested = read_settings(args.config)
    if args.analyze_only:
        cfg = json.loads((args.analyze_only/'manifest.json').read_text())
        if any(cfg[key] != requested[key] for key in ('cap','seed','prefixes','source_run','selection_run')):
            raise ValueError('Cannot change the pre-fixed protocol during recovery')
        cfg['output'] = str(args.analyze_only.resolve())
    else:
        cfg, inputs = prepare(requested)
        cfg = validate_replication(cfg, inputs)
        print('READY: fresh64 draws, fixed cap30, same119002 conditions; NOT new-event holdout', flush=True)
        if args.prepare_only: return
        output = Path(cfg['output_root'])/('confirm-'+uuid.uuid4().hex[:10]); output.mkdir(parents=True)
        cfg['output'] = str(output)
        np.savez(output/'inputs.npz', **inputs); del inputs
        (output/'manifest.json').write_text(json.dumps(cfg, indent=2)+'\n')
    import wandb
    logger = cfg['logger']; output = Path(cfg['output'])
    with wandb.init(entity=logger['entity'], project=logger['project'], name=logger['name'], group=logger['group'],
                    dir=str(output), config=cfg, mode='online',
                    tags=['16-gpu','raw-1110','frozen-FiLM','fixed-cap30','fresh-draws','same-conditions']) as run:
        (output/'wandb.json').write_text(json.dumps(dict(id=run.id,url=run.url))+'\n')
        run.save(str(output/'manifest.json'), base_path=str(output), policy='now')
        run.define_metric('arm_index')
        for metric in ('error', 'error_minus_unweighted', 'event_ess', 'candidate_ess', 'max_candidate_mass'):
            run.define_metric(metric, step_metric='arm_index')
        run.summary.update(dict(phase='sampling', classifier_fits=0, policy_updates=0, reused_samples=0,
                                independent_event_generalization_tested=False))
        try:
            if not args.analyze_only: sample(cfg, run, args.ray_address)
            run.summary['phase'] = 'analysis'
            finish(cfg, run)
        except BaseException:
            run.summary['phase'] = 'failed'
            raise


if __name__ == '__main__': main()
