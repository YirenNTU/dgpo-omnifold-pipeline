"""User-launched residual ratio round + K64 spin endpoint + fresh weighted audits.

Uses existing filtered raw1110 inputs and the saved zrv2yfgt best checkpoint.
No baseline retraining, no generation, no diffusion or DGPO updates. All model
training, feature replay, and inference use 16 GPUs. CPU bootstrap is arithmetic.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts'), str(ROOT/'evenet_dgpo')]

import numpy as np
import torch
import yaml

from scripts import run_tau_condition_width as width
from scripts.run_tau_explicit_inputs import verify_wide, load_saved_scores
from scripts.run_tau_ratio_objectives import merge_scores
from scripts.train_conditional_spin_ratio import build_classifier, score_pair
from scripts.tau_residual_ratio import make_data, compose, log_normalizer, training_worker


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def read_settings(path):
    overlay = yaml.safe_load(Path(path).read_text())
    settings = width.read_settings(ROOT/overlay['width_config'])
    settings.update(overlay)
    if (settings['workers'], settings['batch_size'], settings['cap'], settings['wide_run']) != (16, 1024, 30, 'zrv2yfgt'):
        raise ValueError('Requires the pinned context256 baseline, cumulative cap30 and 16 GPUs x1024')
    t = settings['training']
    if (t['selector'] != 'minimum_validation_weighted_bce' or t['patience'] < 1
            or t['min_delta'] < 0 or t['min_steps'] < 1000 or t['epochs'] < 1
            or not 0 < t['min_lr'] <= t['lr'] or (t['grad_clip'] is not None and t['grad_clip'] <= 0)):
        raise ValueError('Invalid best-validation training/stopping contract')
    name = settings['logger']['name']
    if not isinstance(name, str) or len(name) > 96 or not 3 <= len(name.split(' | ')) <= 5:
        raise ValueError('Invalid W&B display name')
    return settings


def prepare(settings):
    from evenet_dgpo.evenet.dataset.filtered_data import validate_filtered_dataset
    source, old, arrays, _ = width.prepare(settings)
    baseline = verify_wide(settings, old, source, arrays)
    for key, rows in (('train_events', 416701), ('test_events', 119002)):
        if validate_filtered_dataset(baseline[key])['rows'] != rows:
            raise ValueError('Filtered population differs: '+key)
    # No residual training on the base model's gradient-training events.
    keep = arrays['split'] != 0
    keys = ('condition', 'candidate_truth', 'candidate_generated', 'source_ids', 'event_weight', 'split')
    a = {k: arrays[k][keep] for k in keys}
    for old_split, count in ((1, 62213), (2, 119002)):
        if np.sum(a['split'] == old_split) != count:
            raise ValueError('Pinned fit/validation/test identities changed')
    saved = torch.load(Path(settings['wide_directory'])/'best.pt', map_location='cpu', weights_only=True)
    for key in ('epoch', 'optimizer_steps', 'val_bce'):
        if key not in saved:
            raise ValueError('Base best-checkpoint metadata missing: '+key)
    print('READY:', dict(base_run=settings['wide_run'], base_best_epoch=saved['epoch']+1,
          residual_pool=62213, external_evaluation=119002, workers=16, batch_size=1024,
          baseline_refits=0, generated_samples=0, policy_updates=0), flush=True)
    return baseline, a


def architecture(baseline, a):
    keys = ('hidden', 'dropout', 'relative_dim', 'head_kind', 'head_depth', 'condition_hidden',
            'condition_width', 'packing_spec', 'relative_preprocessing')
    cfg = {k: copy.deepcopy(baseline[k]) for k in keys}
    cfg.update(condition_dim=a['condition'].shape[1], candidate_dim=a['candidate_truth'].shape[1],
               ratio_objective='bce', ratio_bound=None)
    return cfg


def fit_config(settings, baseline, a, output, arm, prepared, fit_log_z=None):
    cfg = architecture(baseline, a)
    cfg.update(settings['training'])
    cfg.update(workers=settings['workers'], batch_size=settings['batch_size'], seed=settings['seed'],
               prepared=str(prepared), checkpoint=str(output/arm/'best.pt'), arm=arm,
               base_checkpoint=str(Path(settings['wide_directory'])/'best.pt'),
               base_run=settings['wide_run'], fit_log_z=fit_log_z, cumulative_cap=settings['cap'],
               policy_updates=0, backbone_updates=0, baseline_refits=0,
               ratio_semantics='Balanced weighted BCE: residual logit=log(p/(w1*q/Z1)); accumulated log ratio=s1+s2-log(Z1).',
               checkpoint_selection='Strict minimum validation weighted BCE, from epoch 1; never last or best AUC/Cij.')
    return cfg


@torch.no_grad()
def score_worker(cfg, rank):
    torch.set_num_threads(1)
    device = torch.device('cuda:0')
    saved = torch.load(cfg['checkpoint'], map_location='cpu', weights_only=True)
    model = build_classifier(saved).to(device).eval()
    model.load_state_dict(saved['state_dict'], strict=True)
    with np.load(cfg['score_inputs'], allow_pickle=False) as f:
        pos = np.arange(rank, len(f['source_ids']), cfg['workers'])
        p, q = score_pair(model, f['condition'][pos], f['candidate_truth'][pos],
                         f['candidate_generated'][pos], device, cfg['batch_size'])
        np.savez(Path(cfg['score_output'])/f'score-{rank:02d}.npz', positions=pos,
                 source_ids=f['source_ids'][pos], truth_logits=p, generated_logits=q)


@torch.no_grad()
def panel_worker(cfg, rank):
    """Replay the frozen raw1110 features, then evaluate base AND residual heads."""
    from scripts.tau_fresh_negatives import load_policy, make_batch, candidate_features, isolated_rng
    from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import EventPackingSpec
    torch.set_num_threads(1)
    device = torch.device('cuda:0')
    pos = np.arange(rank, 119002, cfg['workers'])
    panel = Path(cfg['panel_directory'])
    with np.load(panel/'inputs.npz', allow_pickle=False) as f:
        a = {k: f[k][pos] for k in ('raw_condition', 'condition', 'visible_a', 'visible_b', 'source_ids')}
    with np.load(panel/'samples_and_scores.npz', allow_pickle=False) as f:
        delta = f['deltas'][pos]
    with np.load(Path(cfg['wide_directory'])/'fixed_panel_scores.npz', allow_pickle=False) as f:
        if not np.array_equal(f['source_ids'][pos], a['source_ids']):
            raise ValueError('Base K64 source identities differ')
        base_scores = f['logits'][pos]
    base = torch.load(cfg['base_checkpoint'], map_location='cpu', weights_only=True)
    residual = torch.load(cfg['checkpoint'], map_location='cpu', weights_only=True)
    if residual['fit_log_z'] != cfg['fit_log_z'] or residual['base_run'] != cfg['wide_run']:
        raise ValueError('Residual stack provenance differs')
    with isolated_rng(device, cfg['seed']):
        policy = load_policy(cfg, device)
        heads = [build_classifier(saved).to(device).eval() for saved in (base, residual)]
        for head, saved in zip(heads, (base, residual)):
            head.load_state_dict(saved['state_dict'], strict=True)
        spec = EventPackingSpec.from_dict(base['packing_spec'])
        scores = np.empty(delta.shape[:2], dtype=np.float32)
        parity = 0.
        for start in range(0, len(pos), cfg['batch_size']):
            stop = min(start+cfg['batch_size'], len(pos))
            batch = make_batch(a['raw_condition'][start:stop], spec, device)
            c = torch.as_tensor(a['condition'][start:stop], device=device)
            for j in range(delta.shape[1]):
                features = candidate_features(policy, batch, torch.as_tensor(delta[start:stop, j], device=device),
                    a['visible_a'][start:stop], a['visible_b'][start:stop], base['relative_preprocessing'], cfg['feature_batch_size'])
                features = torch.as_tensor(features, device=device)
                if j == 0:
                    replay = heads[0](c, features).cpu().numpy()
                    expected = base_scores[start:stop, j]
                    parity = max(parity, float(np.max(np.abs(replay-expected))))
                    if not np.allclose(replay, expected, atol=1e-3, rtol=1e-4):
                        raise ValueError('Frozen context256 K64 feature/score replay failed')
                scores[start:stop, j] = heads[1](c, features).cpu().numpy()
            print(f'[residual K64 rank={rank}] {stop}/{len(pos)}', flush=True)
    revised = compose(base_scores, scores, cfg['fit_log_z'], cfg['cap'])
    np.savez(Path(cfg['output'])/f'panel-{rank:02d}.npz', positions=pos, source_ids=a['source_ids'],
             logits=revised, residual_logits=scores, max_base_replay_error=parity)


def merge_panel(output, ids, workers):
    score = np.empty((len(ids), 64), dtype=np.float64)
    residual = np.empty_like(score)
    seen = np.zeros(len(ids), int)
    parity = 0.
    for rank in range(workers):
        with np.load(output/f'panel-{rank:02d}.npz', allow_pickle=False) as f:
            pos = f['positions']
            if (pos < 0).any() or (pos >= len(ids)).any() or not np.array_equal(f['source_ids'], ids[pos]):
                raise ValueError('Invalid K64 inference identities')
            np.add.at(seen, pos, 1)
            score[pos], residual[pos] = f['logits'], f['residual_logits']
            parity = max(parity, float(f['max_base_replay_error']))
    if not (seen == 1).all() or not np.isfinite(score).all() or not np.isfinite(residual).all():
        raise ValueError('Missing, repeated or nonfinite K64 residual shard')
    return score, residual, parity


def fit_one(settings, baseline, a, output, arm, prepared, run, fit_log_z=None):
    from ray.train import RunConfig, ScalingConfig, FailureConfig
    from ray.train.torch import TorchTrainer
    from ray.tune import Callback
    cfg = fit_config(settings, baseline, a, output, arm, prepared, fit_log_z)
    n = np.sum(a['split'] == 0)
    possible = int(np.ceil(n/(cfg['workers']*cfg['batch_size'])))*cfg['epochs']
    if possible < cfg['min_steps']:
        raise ValueError('Budget cannot reach minimum fit steps')
    (output/arm).mkdir()
    write_json(output/arm/'config.json', cfg)
    run.define_metric(f'{arm}/epoch')
    run.define_metric(f'{arm}/*', step_metric=f'{arm}/epoch')
    class Progress(Callback):
        def on_trial_result(self, iteration, trials, trial, result, **info):
            keys = ('epoch', 'optimizer_steps', 'train_bce', 'val_bce', 'val_auc', 'best_val_bce',
                    'early_stopped', 'stale_epochs', 'best_saved', 'lr', 'next_lr', 'grad_norm_last')
            run.log({f'{arm}/{key}': result[key] for key in keys if key in result})
    run.summary['phase'] = arm
    TorchTrainer(train_loop_per_worker=training_worker, train_loop_config=cfg,
        scaling_config=ScalingConfig(num_workers=cfg['workers'], use_gpu=True),
        run_config=RunConfig(name=arm, storage_path=str(output/'ray_results'),
            callbacks=[Progress()], failure_config=FailureConfig(max_failures=0))).fit()
    status = json.loads((output/arm/'fit_status.json').read_text())
    run.summary.update({f'{arm}/selected/{key}': value for key, value in status.items()})
    run.save(str(output/arm/'fit_status.json'), base_path=str(output), policy='now')
    return cfg


def run_score(cfg, inputs, output, ids):
    import ray
    output.mkdir()
    job = dict(cfg, score_inputs=str(inputs), score_output=str(output))
    worker = ray.remote(num_gpus=1, num_cpus=1, max_calls=1)(score_worker)
    ray.get([worker.remote(job, rank) for rank in range(cfg['workers'])])
    return merge_scores(output, ids, cfg['workers'])


def audit_comparison(output, data, settings):
    from scripts.audit_conditional_tau_ratio import merge_scores as merge_audit, metrics
    ti = data[0]['split'] == 2
    ids = data[0]['source_ids'][ti]
    endpoints = []
    for arm, a in zip(('audit_baseline', 'audit_residual'), data):
        if not np.array_equal(a['source_ids'], data[0]['source_ids']) or not np.array_equal(a['split'], data[0]['split']):
            raise ValueError('Fresh audit populations differ')
        p, q = merge_audit(output/arm, ids, settings['workers'])
        endpoints.append((p, q, a['event_weight'][ti], a['log_ratio'][ti]))
    def evaluate(idx):
        return [metrics(*[v[idx] for v in endpoint], True) for endpoint in endpoints]
    point = evaluate(np.arange(len(ids)))
    rng = np.random.default_rng(settings['bootstrap_seed'])
    draws = [evaluate(rng.integers(0, len(ids), len(ids))) for _ in range(settings['audit_bootstrap'])]
    comparison = {}
    for key in ('bce', 'auc_gap'):
        values = np.array([[d[i][key] for i in range(2)] for d in draws])
        comparison[key] = dict(change=point[1][key]-point[0][key],
                              ci95=np.quantile(values[:, 1]-values[:, 0], [.025, .975]).tolist())
    statuses = [json.loads((output/arm/'fit_status.json').read_text()) for arm in ('audit_baseline', 'audit_residual')]
    result = dict(baseline=point[0], residual=point[1], residual_minus_baseline=comparison,
        events=len(ids), fit_status=statuses, adequately_trained=all(s['minimum_fit_budget_met'] for s in statuses),
        selected_budget_met=all(s['selected_budget_met'] for s in statuses),
        scope='K1 fresh best-response audit, not K64 classification. Same external 60/20/20 event split and initialization.',
        uncertainty='Paired event bootstrap of two fixed best-val heads; no training uncertainty. AUC gap down and BCE toward log(2) support smaller detectable residual, not full closure.')
    write_json(output/'fresh_audit_report.json', result)
    return result


def physics_report(settings, baseline, output, revised):
    from scripts.tau_moment_balance import paired_report
    panel = Path(settings['panel_directory'])
    with np.load(panel/'inputs.npz', allow_pickle=False) as f:
        inputs = {k: f[k] for k in ('source_ids', 'weight', 'truth_cij', 'category', 'visible_pt_sum')}
    with np.load(panel/'samples_and_scores.npz', allow_pickle=False) as f:
        generated = f['cij']
    _, old, prior = load_saved_scores(settings['wide_directory'], inputs['source_ids'])
    cfg = dict(settings, condition_pt_edges=baseline['condition_pt_edges'])
    report, arrays = paired_report(inputs, generated, old, revised, cfg)
    if not np.allclose(report['arms']['baseline']['C'], prior['arms']['bounded']['C'], atol=1e-9, rtol=1e-8):
        raise ValueError('Baseline K64 Cij did not replay')
    report['arms']['residual'] = report['arms'].pop('calibrated')
    report.update(events=len(old), candidates=64, primary='Residual minus baseline offdiagonal Cij error',
        source_endpoints_verified=True, selection='Checkpoint chosen ONLY by internal validation weighted BCE',
        uncertainty='Paired whole-event bootstrap, fixed heads/candidates; component family9 bands; no fit uncertainty; previously inspected external panel is exploratory.',
        note='Endpoint guardrails are diagnostic. No Cij-based rejection/fallback or checkpoint selection.')
    arrays['arm_names'] = np.array(['unweighted', 'baseline', 'residual'])
    write_json(output/'fixed_K64_report.json', report)
    np.savez(output/'fixed_K64_measurements.npz', **arrays)
    return report


def publish_physics(run, output, physics):
    import wandb
    for arm, row in physics['arms'].items():
        run.summary.update({f'K64/{arm}/{k}': v for k, v in row.items() if isinstance(v, (int, float))})
    for group in ('offdiagonal', 'diagonal', 'total'):
        row = physics['comparisons'][group]
        run.summary.update({f'K64/{group}/change': row['change'], f'K64/{group}/lo95': row['ci95'][0],
                            f'K64/{group}/hi95': row['ci95'][1], f'K64/{group}/status': row['status']})
    run.summary['K64/qualified_improvement'] = physics['qualified_improvement']
    run.log({'K64/components': wandb.Table(columns=['component', 'abs_error_change', 'lo95', 'hi95', 'family_lo95', 'family_hi95'],
        data=[[r['component'], r['change'], *r['pointwise_ci95'], *r['simultaneous_ci95']]
              for r in physics['comparisons']['components']]),
        'K64/condition_groups': wandb.Table(columns=['arm', 'category', 'pt_bin', 'events', 'base_mass', 'weighted_mass', 'error'],
            data=[[arm, *[r[k] for k in ('category', 'pt_bin', 'events', 'base_mass', 'weighted_mass', 'error')]]
                  for arm, row in physics['arms'].items() for r in row['groups']])})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scripts.tau_tail_attribution import AXES
    fig, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    for name, row in physics['arms'].items():
        ax.plot(AXES, row['absolute_component_error'], marker='o', label=name)
    ax.set(title='K64 external Cij closure | frozen best-val residual head', ylabel='Absolute Cij error')
    ax.legend(); fig.savefig(output/'cij_comparison.png', dpi=160); plt.close(fig)
    run.log({'plots/Cij': wandb.Image(str(output/'cij_comparison.png'))})
    for name in ('manifest.json', 'stack.json', 'fixed_K64_report.json', 'cij_comparison.png'):
        run.save(str(output/name), base_path=str(output), policy='now')


def publish_audit(run, output, audit):
    for arm in ('baseline', 'residual'):
        run.summary.update({f'fresh_audit/{arm}/{k}': v for k, v in audit[arm].items()})
    for key, row in audit['residual_minus_baseline'].items():
        run.summary.update({f'fresh_audit/{key}/change': row['change'],
                            f'fresh_audit/{key}/lo95': row['ci95'][0], f'fresh_audit/{key}/hi95': row['ci95'][1]})
    run.summary.update({'fresh_audit/adequately_trained': audit['adequately_trained'],
                        'fresh_audit/selected_budget_met': audit['selected_budget_met']})
    run.save(str(output/'fresh_audit_report.json'), base_path=str(output), policy='now')


def execute(settings, baseline, a, ray_address):
    import ray
    import wandb
    output = Path(settings['output_root'])/('residual-'+uuid.uuid4().hex[:10])
    output.mkdir(parents=True)
    source_inputs = output/'source_inputs.npz'
    np.savez(source_inputs, **a)
    manifest = dict(settings, baseline_run=settings['wide_run'], baseline_directory=settings['wide_directory'],
        upstream_unbounded_run=settings['baseline_run'], upstream_unbounded_directory=settings['baseline_directory'],
        base_best_checkpoint=str(Path(settings['wide_directory'])/'best.pt'),
        backbone_manifest=baseline['backbone_manifest'], train_events=baseline['train_events'], test_events=baseline['test_events'],
        source_split_counts=baseline['split_counts'], residual_split='Old internal validation only, new hash80/20',
        fresh_audit_split='External119002, hash60/20/20; not used to fit/select either ratio stage',
        base_selection_dependence='Base best epoch used this old validation pool; not OOF/untouched. No source gradient-training rows enter residual fitting.',
        representation='Same context256 FiLM depth3 and frozen raw1110 features, no new physics inputs.',
        residual_output='Unbounded correction logits; cap30 applied only to the final cumulative raw ratio.',
        baseline_refits=0, generated_samples=0, policy_updates=0, classifier_fits=3, pristine_test=False)
    write_json(output/'manifest.json', manifest)
    ray.init(address=ray_address, runtime_env={'env_vars': {'PYTHONPATH': os.pathsep.join(
        (str(ROOT), str(ROOT/'scripts'), str(ROOT/'evenet_dgpo'), os.environ.get('PYTHONPATH', '')))}})
    if ray.cluster_resources().get('GPU', 0) < 16:
        ray.shutdown()
        raise ValueError('Requires 16 available cluster GPUs')
    logger = settings['logger']
    try:
        with wandb.init(entity=logger['entity'], project=logger['project'], name=logger['name'], group=logger['group'],
                        config=manifest, dir=str(output), tags=['residual-ratio', 'context256', 'cap30', 'best-val', '16-gpu', 'no-policy-update']) as run:
            write_json(output/'wandb.json', dict(id=run.id, url=run.url))
            try:
                run.summary.update(dict(phase='base_score_replay', baseline_refits=0, generated_samples=0, policy_updates=0))
                run.save(str(output/'manifest.json'), base_path=str(output), policy='now')
                _, base_q = run_score(dict(settings, checkpoint=manifest['base_best_checkpoint']), source_inputs,
                                      output/'base_scores', a['source_ids'])
                ext = a['split'] == 2
                old_q, _, _ = load_saved_scores(settings['wide_directory'], a['source_ids'][ext])
                if not np.allclose(base_q[ext], old_q, atol=1e-3, rtol=1e-4):
                    raise ValueError('Base best checkpoint K1 replay differs from saved scores')
                base_q[ext] = old_q  # keep the exact historical external comparator
                residual_data = make_data(a, base_q, settings['residual_split_seed'])
                fi = residual_data['split'] == 0
                fit_log_z = log_normalizer(residual_data['event_weight'][fi], residual_data['log_ratio'][fi])
                np.savez(output/'residual_inputs.npz', **residual_data)
                run.summary.update({'residual/fit_events': int(fi.sum()), 'residual/val_events': int((~fi).sum()),
                                    'residual/fit_log_Z1': fit_log_z})
                cfg = fit_one(settings, baseline, residual_data, output, 'residual', output/'residual_inputs.npz', run, fit_log_z)
                best = torch.load(cfg['checkpoint'], map_location='cpu', weights_only=True)
                # Freeze the deployable stack BEFORE looking at external physical endpoints.
                stack = dict(base_checkpoint=manifest['base_best_checkpoint'], residual_checkpoint=cfg['checkpoint'],
                    residual_best_epoch=best['epoch'], residual_best_steps=best['optimizer_steps'],
                    residual_best_val_bce=best['val_bce'], fit_log_z=fit_log_z, cap=settings['cap'],
                    formula='min(log(cap), base_log_ratio + residual_logit - fit_log_z)',
                    selection='minimum_validation_weighted_bce', test_used_for_selection=False)
                write_json(output/'stack.json', stack)
                np.savez(output/'external_inputs.npz', **{k: v[ext] for k, v in a.items()})
                run.summary['phase'] = 'best_residual_K1_inference'
                p2, q2 = run_score(cfg, output/'external_inputs.npz', output/'residual_scores', a['source_ids'][ext])
                revised_k1 = compose(old_q, q2, fit_log_z, settings['cap'])
                np.savez(output/'test_scores.npz', source_ids=a['source_ids'][ext], baseline_log_ratio=old_q,
                         residual_truth_logits=p2, residual_generated_logits=q2, log_ratio=revised_k1)
                run.summary['phase'] = 'best_residual_K64_inference'
                panel_cfg = dict(settings, output=str(output), checkpoint=cfg['checkpoint'], base_checkpoint=manifest['base_best_checkpoint'],
                    fit_log_z=fit_log_z, fresh_runtime=baseline['backbone_manifest']['runtime'],
                    fresh_generator_checkpoint=baseline['backbone_manifest']['checkpoint'])
                worker = ray.remote(num_gpus=1, num_cpus=1, max_calls=1)(panel_worker)
                ray.get([worker.remote(panel_cfg, rank) for rank in range(16)])
                revised, residual_scores, parity = merge_panel(output, a['source_ids'][ext], 16)
                np.savez(output/'fixed_panel_scores.npz', source_ids=a['source_ids'][ext], logits=revised, residual_logits=residual_scores)
                run.summary['base/max_K64_replay_error'] = parity
                from scripts.tau_tail_attribution import candidate_weights
                _, base_panel, _ = load_saved_scores(settings['wide_directory'], a['source_ids'][ext])
                uncapped = base_panel + residual_scores - fit_log_z
                tail = {}
                for label, logits in (('baseline', base_panel), ('uncapped_product', uncapped), ('capped_product', revised)):
                    w, logmean = candidate_weights(a['event_weight'][ext], logits)
                    tail[label] = dict(candidate_ess_fraction=float(1/np.square(w).sum()/w.size),
                        event_ess_fraction=float(1/np.square(w.sum(1)).sum()/len(w)),
                        max_candidate_mass=float(w.max()), log_mean_ratio=logmean,
                        log_ratio_max=float(logits.max()), cap_fraction=float(np.mean(logits >= np.log(settings['cap']))))
                    run.summary.update({f'product_health/{label}/{k}': v for k, v in tail[label].items()})
                write_json(output/'product_health.json', tail)
                run.save(str(output/'product_health.json'), base_path=str(output), policy='now')
                run.summary['phase'] = 'paired_K64_spin_evaluation'
                physics = physics_report(settings, baseline, output, revised)
                publish_physics(run, output, physics)
                audits = []
                for arm, external_score in (('audit_baseline', old_q), ('audit_residual', revised_k1)):
                    full_score = base_q.copy(); full_score[ext] = external_score
                    data = make_data(a, full_score, settings['audit_split_seed'], audit=True)
                    path = output/(arm+'_inputs.npz'); np.savez(path, **data)
                    fit_one(settings, baseline, data, output, arm, path, run)
                    audits.append(data)
                audit = audit_comparison(output, audits, settings)
                publish_audit(run, output, audit)
                write_json(output/'COMPLETE', dict(run_id=run.id, all_endpoints_complete=True))
                write_json(Path(settings['output_root'])/'latest_completed.json', dict(output=str(output), run_id=run.id, url=run.url))
                run.summary.update(dict(phase='complete', report_path=str(output/'fixed_K64_report.json')))
                print('OUTPUT:', output, 'WANDB:', run.url, flush=True)
            except BaseException:
                run.summary['phase'] = 'failed'
                raise
    finally:
        ray.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('phase', choices=('prepare', 'train'), nargs='?', default='train')
    parser.add_argument('--ray-address', default=os.environ.get('RAY_ADDRESS') or 'auto')
    args = parser.parse_args()
    settings = read_settings(args.config)
    baseline, arrays = prepare(settings)
    if args.phase == 'train':
        execute(settings, baseline, arrays, args.ray_address)


if __name__ == '__main__':
    main()
