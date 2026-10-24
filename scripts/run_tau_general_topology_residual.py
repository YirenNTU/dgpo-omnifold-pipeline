"""User-launched classifier-only step500 General/topology correction experiment.

check: config only; verify: read-only sources; prepare: snapshot fixed inputs;
run: fit only two classifier arms on 16 GPUs, score saved candidates and report.
Never generates diffusion candidates, updates a policy or submits a job.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts'), str(ROOT/'evenet_dgpo')]
import numpy as np
import torch
import yaml

from scripts.run_tau_physics_branch import load_panel, merge_candidates
from scripts.tau_general_topology_residual import ARMS, PAIRS, PRIMARY, WARM_CORRECTIONS, build_arm, fit_arm, full_truth_comparison
from scripts.tau_topology_branch import topology_features
from scripts.train_diffusion_film_10pct import TRAIN, VAL

TRAIN_KEYS = ('condition', 'candidate_truth', 'candidate_generated', 'event_weight', 'split', 'source_ids')


def correction_kind(cfg):
    return cfg.get('correction', 'topology')


def uses_cached_controls(cfg):
    return correction_kind(cfg) in (*WARM_CORRECTIONS, 'pair_interaction')


def training_arms(cfg):
    return ('residual',) if uses_cached_controls(cfg) else ('continued', 'residual')


def experiment_primary(cfg):
    if correction_kind(cfg) == 'cross_attention':
        from scripts.tau_cross_attention_experiment import PRIMARY as attention_primary
        return attention_primary
    if correction_kind(cfg) == 'fine_angles':
        from scripts.tau_fine_angle_residual import PRIMARY as fine_primary
        return fine_primary
    return PRIMARY


def correction_features(cfg, visible_a, visible_b, deltas):
    if correction_kind(cfg) == 'pair_interaction':
        from scripts.tau_pair_interaction import pair_features
        return pair_features(visible_a, visible_b, deltas)
    if uses_cached_controls(cfg):
        # Warm head variants use the existing inputs without extra target features.
        return np.empty((len(deltas), 0), dtype=np.float32)
    if correction_kind(cfg) == 'fine_angles':
        from scripts.tau_fine_angle_residual import fine_angle_features
        return fine_angle_features(visible_a, visible_b, deltas)
    return topology_features(visible_a, visible_b, deltas)


def settings(path):
    cfg = yaml.safe_load(Path(path).read_text())
    if correction_kind(cfg) not in ('topology', 'fine_angles', 'pair_interaction', *WARM_CORRECTIONS):
        raise ValueError('Unknown classifier correction')
    if correction_kind(cfg) == 'pair_interaction':
        from scripts.tau_pair_interaction import feature_spec
        if cfg.get('pair_spec') != feature_spec():
            raise ValueError('Pair-interaction architecture differs from declaration')
    if correction_kind(cfg) == 'cross_attention':
        from scripts.tau_cross_attention_experiment import attention_spec
        if cfg.get('attention_spec') != attention_spec():
            raise ValueError('Attention architecture differs from declared experiment')
    if correction_kind(cfg) == 'film_mlp':
        from scripts.tau_film_mlp import film_mlp_spec
        if cfg.get('film_mlp_spec') != film_mlp_spec():
            raise ValueError('FiLM MLP architecture differs from declared experiment')
    if uses_cached_controls(cfg):
        if cfg.get('reuse_controls') is not True or cfg.get('control_run') != '9kqijsch' or not cfg.get('control_directory'):
            raise ValueError('Warm-head experiment must reuse completed 9kqijsch controls')
    if correction_kind(cfg) == 'fine_angles':
        from scripts.tau_fine_angle_residual import feature_spec
        if cfg.get('feature_spec') != feature_spec():
            raise ValueError('Fine-angle feature schema differs from declared experiment')
    required = dict(source_run='c34035e0', policy_step=500, workers=16, validation_events=119002,
                    validation_candidates=8, ratio_cap=30, seed=42)
    if any(cfg.get(k) != v for k, v in required.items()) or cfg['bootstrap'] < 1000:
        raise ValueError('Requires fixed step500, filtered full validation, K8, cap30 and 16 GPUs')
    f = cfg['fit']
    if (f['batch_size'] != 1024 or f['epochs'] != 250 or f['min_selected_steps'] < 1000
            or f['epochs']*22 < f['min_selected_steps'] or not 0 < f['min_lr'] <= f['lr']
            or f['weight_decay'] < 0):
        raise ValueError('Use the declared common 5500-update continuation/correction budget')
    src, out = Path(cfg['source_directory']).resolve(), Path(cfg['output']).resolve()
    if out.is_relative_to(src) or src.is_relative_to(out):
        raise ValueError('Source and experiment output must be separate')
    name = cfg['logger']['name']
    if not name.isascii() or len(name) > 96 or len(name.split(' | ')) < 3 or '_' in name:
        raise ValueError('Use a readable ASCII W&B name')
    return cfg


def verify(cfg):
    from evenet.dataset.filtered_data import validate_filtered_dataset
    from RL.DGPO_neutrino.conditional_tau_reward import validate_head
    root = Path(cfg['source_directory'])
    runtime = yaml.safe_load((root/'runtime.yaml').read_text())
    for key, path, count in [('data_parquet_dir', TRAIN, 416701), ('data_parquet_val_dir', VAL, 119002)]:
        if runtime['platform'][key] != path or validate_filtered_dataset(path)['rows'] != count:
            raise ValueError('Wrong or incomplete filtered input')
    tau = runtime['dgpo']['tau_ratio']
    if (tau['policy_conditioning_contract'] != 'supervised_process_labels_v2'
            or tau['source_global_step'] != 18655 or tau['source_completed_epochs'] != 287):
        raise ValueError('Wrong FiLM epoch287 backbone/conditioning provenance')
    ckpt = torch.load(cfg['policy_checkpoint'], map_location='cpu', weights_only=False, mmap=True)
    if ckpt['global_step'] != 500:
        raise ValueError('Wrong policy checkpoint')
    del ckpt
    training = root/'tau_diagnostics/refit-step-00000500'
    validation = root/'tau_diagnostics/eval-step-00000500'
    report = json.loads((validation/'report.json').read_text())
    status = json.loads((training/'best.json').read_text())
    if (report['policy_step'], report['events'], report['candidates']) != (500, 119002, 8):
        raise ValueError('Wrong saved validation endpoint')
    if status['best_steps'] < 1000 or not status['minimum_fit_steps_met']:
        raise ValueError('Incomplete/undertrained source General')
    head = torch.load(training/'best.pt', map_location='cpu', weights_only=True)
    validate_head(head)
    if (head['source_policy_step'] != 500 or head['candidate_dim'] != 533 or head['condition_dim'] != 531
            or head['num_process_classes'] != 19 or head['dropout'] != 0. or head['ratio_bound'] != 30):
        raise ValueError('Wrong fitted General schema/denominator')
    bundle = torch.load(root/'initial_tau_reward.pt', map_location='cpu', weights_only=True)
    for key in ('source_checkpoint', 'normalization_file', 'backbone_runtime'):
        if not Path(bundle[key]).is_file():
            raise ValueError('Missing pinned feature input: '+key)
    for key in ('packing_spec', 'relative_preprocessing', 'policy_conditioning_contract'):
        if head[key] != bundle['head'][key]:
            raise ValueError('Source General and cached feature schema differ: '+key)
    import pyarrow.parquet as pq
    from scripts.diagnose_ztautau_cij import SOURCE_ID_COLUMNS
    files = sorted(Path(VAL).glob('*.parquet'))
    needed = set(SOURCE_ID_COLUMNS) | {p+'_'+k for p in
        ('truth_tau_a', 'truth_tau_b', 'truth_a_visible', 'truth_b_visible') for k in ('E', 'px', 'py', 'pz')}
    if not files or any(not needed.issubset(pq.ParquetFile(f).schema_arrow.names) for f in files):
        raise ValueError('Complete-truth reporting columns missing; no reference fallback')
    for folder, n, k in [(training, 416701, 1), (validation, 119002, 8)]:
        for rank in range(16):
            with np.load(folder/f'candidates/rank-{rank:02d}.npz', allow_pickle=False) as f:
                pos = np.arange(rank, n, 16)
                np.testing.assert_array_equal(f['positions'], pos)
                if f['deltas'].shape != (len(pos), k, 2, 2) or f['features'].shape != (len(pos), 533):
                    raise ValueError('Wrong cached candidate layout')
    # No neural forward in verify. Check exact fitted tensor inheritance and
    # which parameters will be trainable in the correction arm.
    residual = build_arm(head, 'residual', correction=correction_kind(cfg))
    inherited = residual if correction_kind(cfg) in WARM_CORRECTIONS else residual.base
    for key, value in head['state_dict'].items():
        if not torch.equal(value, inherited.state_dict()[key]):
            raise ValueError('General initialization differs')
    if correction_kind(cfg) not in WARM_CORRECTIONS and any(p.requires_grad for p in inherited.parameters()):
        raise ValueError('Residual arm must freeze General')
    if uses_cached_controls(cfg):
        from scripts.tau_cross_attention_experiment import verify_controls
        verify_controls(cfg, head)
    print('SOURCE OK: completed step500 refit/eval, filtered manifests, actual labels, exact General inheritance, cap30; correction='+correction_kind(cfg), flush=True)
    return root, training, validation, head, report, bundle


def complete_truth_terms(panel):
    """Truth p4 is used ONLY for reporting, never in classifier features."""
    from scripts.diagnose_ztautau_cij import read_event_table, source_ids, align, SOURCE_ID_COLUMNS, angles
    prefixes = ('truth_tau_a', 'truth_tau_b', 'truth_a_visible', 'truth_b_visible',
                'lead_a_visible', 'lead_b_visible')
    columns = list(SOURCE_ID_COLUMNS)+['event_weight', 'analyzing_power_a', 'analyzing_power_b', 'event_category']
    columns += [p+'_'+k for p in prefixes for k in ('E', 'px', 'py', 'pz')]
    table = read_event_table(VAL, columns)
    data = {k: np.asarray(table[k].to_numpy()) for k in columns}
    order = align(source_ids(data), panel['source_ids'])
    data = {k: v[order] for k, v in data.items()}
    np.testing.assert_array_equal(data['event_weight'], panel['event_weight'])
    np.testing.assert_array_equal(data['event_category'], panel['category'])
    kappa = np.stack([data['analyzing_power_'+a] for a in ('a', 'b')], -1)
    np.testing.assert_array_equal(kappa, panel['kappas'])
    p4 = [np.stack([data[p+'_'+k] for k in ('E', 'px', 'py', 'pz')], -1).astype(float) for p in prefixes]
    ta, tb, va, vb, ra, rb = p4
    np.testing.assert_array_equal(ra, panel['visible_a'])
    np.testing.assert_array_equal(rb, panel['visible_b'])
    expected = []
    for t, v in ((ta, ra), (tb, rb)):
        theta = lambda p: np.arctan2(np.linalg.norm(p[:, 1:3], axis=1), p[:, 3])
        phi = lambda p: np.arctan2(p[:, 2], p[:, 1])
        expected.append(np.stack((theta(t)-theta(v), phi(t)-phi(v)), -1))
    difference = panel['truth_deltas']-np.stack(expected, 1)
    difference[..., 1] = np.arctan2(np.sin(difference[..., 1]), np.cos(difference[..., 1]))
    if not np.isfinite(difference).all() or np.max(np.abs(difference)) > 1e-4:
        raise ValueError('Truth direction/normalization/leg mapping differs')
    a, b = angles(ta, tb, va, vb)
    return 9*(a[:, :, None]*b[:, None, :]).reshape(-1, 9)/kappa.prod(1)[:, None]


def prepare(cfg, verified):
    root, training, validation, head, report, original_bundle = verified
    out = Path(cfg['output'])
    if (out/'PREPARED').is_file():
        if json.loads((out/'settings.json').read_text()) != cfg:
            raise ValueError('Prepared settings differ')
        return out
    if out.exists() and any(out.iterdir()):
        raise ValueError('Partial output exists; choose a new readable sequence number')
    out.mkdir(parents=True)
    (out/'settings.json').write_text(json.dumps(cfg, indent=2)+'\n')
    shutil.copy2(training/'best.pt', out/'source_general.pt')
    shutil.copy2(training/'best.json', out/'source_general_fit.json')
    shutil.copy2(validation/'report.json', out/'source_validation_report.json')
    shutil.copy2(root/'runtime.yaml', out/'source_actor_runtime.yaml')
    if uses_cached_controls(cfg):
        control = Path(cfg['control_directory'])
        (out/'continued').mkdir()
        shutil.copy2(control/'continued/best.json', out/'continued/best.json')
        shutil.copy2(control/'continued/best.pt', out/'continued/best.pt')
        for rank in range(16):
            shutil.copy2(control/f'scores-rank-{rank:02d}.npz', out/f'control-scores-rank-{rank:02d}.npz')
        (out/'control_provenance.json').write_text(json.dumps(dict(run=cfg['control_run'],
            directory=str(control), reused_arms=['general','continued'], retrained=False), indent=2)+'\n')
    bundle = copy.deepcopy(original_bundle)
    for key, name in [('source_checkpoint', 'feature_source.ckpt'), ('normalization_file', 'normalization.pt')]:
        shutil.copy2(bundle[key], out/name)
        bundle[key] = str(out/name)
    runtime = yaml.safe_load(Path(bundle['backbone_runtime']).read_text())
    runtime['options']['Dataset']['normalization_file'] = str(out/'normalization.pt')
    (out/'feature_runtime.yaml').write_text(yaml.safe_dump(runtime, sort_keys=False))
    bundle['backbone_runtime'] = str(out/'feature_runtime.yaml')
    torch.save(bundle, out/'feature_bundle.pt')
    panels = {a: load_panel(root/(a+'_panel.npz')) for a in ('train', 'validation')}
    if np.intersect1d(panels['train']['source_ids'], panels['validation']['source_ids']).size:
        raise ValueError('Train/external event overlap')
    from RL.DGPO_neutrino.conditional_tau_reward import condition_inputs
    from scripts.tau_physics_branch_report import terms
    for label, n, folder, k in [('train', 416701, training, 1), ('validation', 119002, validation, 8)]:
        p = panels[label]
        if len(p['source_ids']) != n or len(np.unique(p['source_ids'])) != n or not bool(p['truth_features_ready']):
            raise ValueError('Incomplete/duplicate panel')
        if p['candidate_truth'].shape != (n, 533) or not np.isfinite(p['candidate_truth']).all():
            raise ValueError('Wrong truth feature cache')
        for start in range(0, n, 2048):
            sl = slice(start, start+2048)
            c = condition_inputs(p['raw_condition'][sl], p['category'][sl], bundle['condition_mean'],
                                 bundle['condition_scale'], head['packing_spec'])
            np.testing.assert_array_equal(c, p['condition'][sl])
        d, f = merge_candidates(folder/'candidates', n, k)
        if label == 'train':
            if (np.count_nonzero(p['split'] == 0), np.count_nonzero(p['split'] == 1)) != (354488, 62213):
                raise ValueError('Fitting/selection split changed')
            arrays = dict(condition=p['condition'], event_weight=p['event_weight'], split=p['split'], source_ids=p['source_ids'])
            for name, x, delta in [('truth', p['candidate_truth'], p['truth_deltas']), ('generated', f, d[:, 0])]:
                arrays['candidate_'+name] = np.c_[x, correction_features(cfg, p['visible_a'], p['visible_b'], delta)]
            for key, value in arrays.items():
                np.save(out/('train_'+key+'.npy'), value)
        else:
            t = terms(p, p['truth_deltas'])
            g = np.mean([terms(p, d[:, j]) for j in range(k)], axis=0)
            with np.load(validation/'measurements.npz', allow_pickle=False) as m:
                for key, value in [('source_ids', p['source_ids']), ('weight', p['event_weight']), ('deltas', d)]:
                    np.testing.assert_array_equal(m[key], value)
                np.testing.assert_allclose(t, m['truth_cij'], rtol=0, atol=1e-9)
                np.testing.assert_allclose(g, m['generated_cij'], rtol=0, atol=1e-9)
            for value, key in [(t, 'truth'), (g, 'generated')]:
                np.testing.assert_allclose(np.average(value, weights=p['event_weight'], axis=0).reshape(3, 3),
                                           report[key], rtol=0, atol=1e-9)
            np.save(out/'validation_deltas.npy', d)
            np.save(out/'validation_candidate0.npy', f)
            np.savez(out/'validation_panel.npz', **p)
            np.save(out/'validation_full_truth.npy', complete_truth_terms(p))
    if uses_cached_controls(cfg):
        from scripts.tau_cross_attention_experiment import verify_prepared_controls
        verify_prepared_controls(out, cfg['control_directory'], candidate_prefix=533 if correction_kind(cfg)=='pair_interaction' else None)
    (out/'PREPARED').write_text('Source General, fixed step500 pairs, normalizers, matched and full-truth reporting inputs verified.\n')
    return out


def finish(cfg, run):
    from scripts.tau_physics_branch_report import write_report
    from scripts.train_conditional_spin_ratio import pair_metrics
    out = Path(cfg['output'])
    panel = load_panel(out/'validation_panel.npz')
    delta = np.load(out/'validation_deltas.npy', mmap_mode='r')
    n = len(delta)
    scores = {a: np.empty((n, 8), np.float32) for a in ARMS}
    positives = {a: np.empty(n, np.float32) for a in ARMS}
    seen = np.zeros(n, int)
    for rank in range(16):
        with np.load(out/f'scores-rank-{rank:02d}.npz', allow_pickle=False) as f:
            pos = f['positions']
            np.testing.assert_array_equal(pos, np.arange(rank, n, 16))
            np.testing.assert_array_equal(f['source_ids'], panel['source_ids'][pos])
            np.add.at(seen, pos, 1)
            for arm in ARMS:
                scores[arm][pos], positives[arm][pos] = f[arm], f[arm+'_truth']
    if not np.all(seen == 1) or any(not np.isfinite(v).all() for v in [*scores.values(), *positives.values()]):
        raise ValueError('Incomplete/nonfinite scoring')
    np.savez(out/'scores.npz', source_ids=panel['source_ids'], **scores, **{a+'_truth': v for a, v in positives.items()})
    report, plots = write_report(out, panel, delta, scores, cfg['bootstrap'], cfg['bootstrap_seed'],
                                comparison_pairs=PAIRS, primary='Secondary matched-reference comparison; full-truth primary reported separately')
    full = full_truth_comparison(panel, delta, scores, np.load(out/'validation_full_truth.npy'),
                                 cfg['bootstrap'], cfg['bootstrap_seed'])
    for a in report['arms']:
        np.testing.assert_allclose(report['arms'][a]['C'], full['arms'][a]['C'], rtol=0, atol=1e-10)
    report['matched_primary_note'] = report.pop('primary')
    report.update(primary=experiment_primary(cfg), full_truth=full,
                  decision=dict(primary_pass=full['primary_pass'], automatic_policy_update=False,
                                follow_up='Review distribution, channel-mass and ESS tradeoffs; not a deployment gate.'),
                  provenance=dict(source_run=cfg['source_run'], policy_step=500, policy_checkpoint=cfg['policy_checkpoint'],
                                  source_general='source_general.pt', policy_updates=0, candidate_generation=False))
    test = {a: pair_metrics(positives[a], scores[a][:, 0], panel['event_weight']) for a in ARMS}
    report['heldout_classifier'] = test
    report['fit'] = {a: json.loads((out/a/'best.json').read_text()) for a in ('continued', 'residual')}
    if uses_cached_controls(cfg):
        report['provenance']['reused_controls'] = json.loads((out/'control_provenance.json').read_text())
    if report['fit']['continued']['total_steps'] != report['fit']['residual']['total_steps']:
        raise ValueError('Unmatched additional budgets')
    report['correction'] = dict(score_delta_mean=float((scores['residual']-scores['general']).mean()),
                               score_delta_rms=float(np.sqrt(np.mean((scores['residual']-scores['general'])**2))))
    if correction_kind(cfg) in ('fine_angles', 'pair_interaction', *WARM_CORRECTIONS):
        from scripts.tau_fine_angle_report import write_fine_report
        source = torch.load(out/'source_general.pt', map_location='cpu', weights_only=True)
        process_index = build_arm(source, 'general').process_label_index
        labels = panel['condition'][:, process_index]
        if not np.array_equal(labels, labels.astype(int)):
            raise ValueError('Noninteger process labels in external report')
        fine, fine_plots = write_fine_report(out, panel, delta, scores, positives, cfg['bootstrap'],
                                            cfg['bootstrap_seed'], labels.astype(int))
        report['fine_angles'] = fine
        report['decision'].update(primary_pass=fine['primary_pass'], joint_guard_pass=fine['joint_guard']['passed'])
        report['full_truth']['endpoint_role'] = 'Secondary for this '+correction_kind(cfg)+' experiment; original topology experiment endpoint unchanged'
        plots.extend(fine_plots)
        if correction_kind(cfg) == 'cross_attention':
            from scripts.tau_cross_attention_experiment import bce_comparison
            attention = bce_comparison(positives, scores, panel['event_weight'], cfg['bootstrap'], cfg['bootstrap_seed'])
            report['attention'] = attention
            fine['endpoint_role'] = 'Secondary angular diagnostic; attention primary is external paired BCE'
            (out/'fine_angle_report.json').write_text(json.dumps(fine, indent=2, allow_nan=False)+'\n')
            (out/'attention_report.json').write_text(json.dumps(attention, indent=2, allow_nan=False)+'\n')
            report['decision'].update(primary_pass=attention['primary_pass'],
                follow_up='Review held-out reweighting, Cij and ESS; concurrence evaluation is separate; no automatic DGPO deployment.')
            report['arm_labels'] = dict(general='Source General', continued='Continued General', residual='General plus cross-attention')
        elif correction_kind(cfg) in ('film_mlp','pair_interaction'):
            report['decision'].update(primary_pass=full['primary_pass'],
                follow_up='Primary is full-truth Cij improvement versus both cached controls; inspect joint distributions and ESS. No automatic policy update.')
            report['full_truth']['endpoint_role'] = 'Primary: full-truth-reference Cij distance'
            fine['endpoint_role'] = 'Secondary angular diagnostic; primary is full-truth Cij'
            (out/'fine_angle_report.json').write_text(json.dumps(fine,indent=2,allow_nan=False)+'\n')
            report['arm_labels'] = dict(general='Source General',continued='Continued General',residual='General plus pair interactions' if correction_kind(cfg)=='pair_interaction' else 'General plus FiLM MLP')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(14, 3.8), layout='constrained')
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 4)
    names = list(full['arms'])
    differences = [np.array(full['arms'][a]['C'])-full['truth_C'] for a in names]
    scale = max(float(np.abs(d).max()) for d in differences) or 1.
    for ax, a, d in zip(axes, names, differences):
        im = ax.imshow(d, vmin=-scale, vmax=scale, cmap='coolwarm')
        ax.set_title(a+' minus full truth')
        ax.set_xticks(range(3), list('krn')); ax.set_yticks(range(3), list('krn'))
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f'{d[i,j]:.3f}', ha='center', va='center')
        fig.colorbar(im, ax=ax)
    full_plot = out/'full_truth_cij_errors.png'
    fig.savefig(full_plot, dpi=140)
    plots.append(full_plot)
    (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    (out/'full_truth_report.json').write_text(json.dumps(full, indent=2, allow_nan=False)+'\n')
    import wandb
    run.summary.update({'phase': 'complete', 'primary_pass': report['decision']['primary_pass'], 'policy_updates': 0})
    if correction_kind(cfg) == 'cross_attention':
        for a,row in attention['arms'].items():
            run.summary[a+'/attention/paired_bce_k8'] = row['bce']
        for name,row in attention['comparisons'].items():
            run.summary.update({'attention/'+name+'/delta':row['delta'],
                'attention/'+name+'/ci_low':row['ci95'][0], 'attention/'+name+'/ci_high':row['ci95'][1]})
    if correction_kind(cfg) in ('fine_angles', 'pair_interaction', *WARM_CORRECTIONS):
        run.summary['fine_angles/joint_guard_pass'] = fine['joint_guard']['passed']
        for a, row in fine['arms'].items():
            run.summary[a+'/fine_angles/core_error'] = row['error']
            for i, leg in enumerate(('a','b')):
                run.summary[a+'/fine_angles/'+leg+'_phi_core_mass'] = row['core_mass'][i]
        for pair,row in fine['comparisons'].items():
            run.summary.update({'fine_angles/'+pair+'/delta':row['delta'],
                                'fine_angles/'+pair+'/ci_low':row['ci95'][0], 'fine_angles/'+pair+'/ci_high':row['ci95'][1]})
        for a,row in fine['calibration'].items():
            run.summary.update({a+'/calibration/'+k:row[k] for k in ('ece','brier','negative_mean_ratio')})
        for name,row in fine['joint_guard']['residual_minus_control_tv'].items():
            run.summary.update({'fine_angles/joint/'+name+'/minus_'+a:value for a,value in row.items()})
    for a in ARMS:
        run.summary.update({a+'/test/'+key: value for key, value in test[a].items()})
    for a, values in report['arms'].items():
        run.summary.update({a+'/matched/'+key: value for key, value in values.items() if isinstance(value, (float, int))})
        run.summary[a+'/full_truth/error'] = full['arms'][a]['error']
    for name, row in full['comparisons'].items():
        run.summary.update({'full_truth/'+name+'/delta': row['delta'], 'full_truth/'+name+'/ci_low': row['ci95'][0],
                            'full_truth/'+name+'/ci_high': row['ci95'][1]})
    for name, arms in report['distributions'].items():
        for a, values in arms.items():
            run.summary.update({a+'/distribution/'+name+'/'+key: value for key, value in values.items()})
    run.log({'plots/'+p.stem: wandb.Image(str(p)) for p in plots})
    for name in ('report.json', 'full_truth_report.json', 'histograms.json', 'settings.json', 'source_general_fit.json'):
        run.save(str(out/name), base_path=str(out), policy='now')
    if correction_kind(cfg) == 'cross_attention':
        run.save(str(out/'attention_report.json'), base_path=str(out), policy='now')
    if uses_cached_controls(cfg):
        run.save(str(out/'control_provenance.json'), base_path=str(out), policy='now')
    if correction_kind(cfg) in ('fine_angles', 'pair_interaction', *WARM_CORRECTIONS):
        for name in ('fine_angle_report.json','fine_angle_histograms.json'):
            run.save(str(out/name), base_path=str(out), policy='now')
    (out/'COMPLETE').write_text('Classifier-only comparison completed; no diffusion or DGPO update.\n')
    return report


def worker(cfg):
    import ray.train
    from ray.train.torch import get_device
    from RL.DGPO_neutrino.conditional_tau_cycle import panel_batch, barrier
    from RL.DGPO_neutrino.conditional_tau_reward import load_reward, feature_precision
    ctx = ray.train.get_context()
    rank, world, device = ctx.get_world_rank(), ctx.get_world_size(), get_device()
    if world != 16 or device.type != 'cuda':
        raise ValueError('Requires 16 GPU workers')
    torch.set_num_threads(1)
    out = Path(cfg['output'])
    source = torch.load(out/'source_general.pt', map_location='cpu', weights_only=True)
    arrays = {k: np.load(out/('train_'+k+'.npy'), mmap_mode='r') for k in TRAIN_KEYS}
    run = None
    if rank == 0:
        import wandb
        run = wandb.init(**cfg['logger'], dir=str(out), config={**cfg, 'primary': experiment_primary(cfg), 'policy_updates': 0,
                         'candidate_generation': False, 'arms': list(ARMS), 'source_general_updates': source['optimizer_steps']},
                         tags=['classifier-only', 'warm-general' if correction_kind(cfg) in WARM_CORRECTIONS else 'frozen-general',
                               correction_kind(cfg).replace('_','-'), 'step500', 'cap30', '16-gpu'])
        (out/'wandb.json').write_text(json.dumps(dict(id=run.id, url=run.url)))
        for a in ('continued', 'residual'):
            run.define_metric(a+'/*', step_metric=a+'/optimizer_steps')
    try:
        baseline = build_arm(source, 'general').to(device)
        probe = build_arm(source, 'residual', correction=correction_kind(cfg)).to(device)
        take = np.arange(rank, min(len(arrays['condition']), world*8), world)
        c = torch.tensor(arrays['condition'][take], device=device)
        x = torch.tensor(arrays['candidate_generated'][take], device=device)
        with torch.no_grad(), feature_precision():
            torch.testing.assert_close(baseline(c, x[:, :533]), probe(c, x), rtol=0, atol=0)
        from scripts.train_conditional_spin_ratio import score_pair, pair_metrics
        import torch.distributed as dist
        vi = np.flatnonzero(arrays['split'] == 1)[rank::world]
        with feature_precision():
            p, q = score_pair(baseline, arrays['condition'][vi], arrays['candidate_truth'][vi, :533],
                              arrays['candidate_generated'][vi, :533], device, cfg['fit']['batch_size'])
        gathered = [None]*world if rank == 0 else None
        dist.gather_object((vi, p, q), gathered, dst=0)
        if rank == 0:
            ids = np.concatenate([r[0] for r in gathered])
            np.testing.assert_array_equal(np.sort(ids), np.flatnonzero(arrays['split'] == 1))
            m = pair_metrics(np.concatenate([r[1] for r in gathered]), np.concatenate([r[2] for r in gathered]),
                             arrays['event_weight'][ids])
            np.testing.assert_allclose(m['bce'], source['val_bce'], rtol=0, atol=1e-5)
            run.summary.update({'initial_general_replay_verified': True, 'initial_residual_parity_verified': True,
                                **{'general/selection/'+k: v for k, v in m.items()}})
        barrier(world)
        del baseline, probe, c, x, gathered
        trained_arms = training_arms(cfg)
        if rank == 0 and uses_cached_controls(cfg):
            run.summary.update({'controls/reused': True, 'controls/source_run': cfg['control_run'],
                                'controls/new_optimizer_updates': 0})
        for arm in trained_arms:
            if rank == 0:
                run.summary['phase'] = 'fitting_'+arm
            a = dict(arrays)
            if arm == 'continued':
                for key in ('candidate_truth', 'candidate_generated'):
                    a[key] = arrays[key][:, :533]
            def emit(row):
                run.log({arm+'/'+k: v for k, v in row.items()})
                print(f"[{arm}] updates={row['optimizer_steps']} val_bce={row['val_bce']:.6f}", flush=True)
            with feature_precision():
                model, status = fit_arm(source, arm, cfg['fit'], a, out/arm/'best.pt', device, rank, world, emit,
                                        seed=cfg['seed'], correction=correction_kind(cfg))
            del model, a
            if rank == 0:
                run.summary.update({arm+'/fit/'+k: v for k, v in status.items()})
        del arrays
        panel = load_panel(out/'validation_panel.npz')
        deltas = np.load(out/'validation_deltas.npy', mmap_mode='r')
        candidate0 = np.load(out/'validation_candidate0.npy', mmap_mode='r')
        models = {} if uses_cached_controls(cfg) else {'general': build_arm(source, 'general').to(device)}
        for a in trained_arms:
            saved = torch.load(out/a/'best.pt', map_location='cpu', weights_only=True)
            if saved.get('correction','topology') != correction_kind(cfg):
                raise ValueError('Scoring checkpoint correction type differs')
            models[a] = build_arm(source, a, state=saved['state_dict'], correction=correction_kind(cfg)).to(device).eval().requires_grad_(False)
        reward = load_reward(out/'feature_bundle.pt', device)
        idx = np.arange(rank, len(deltas), world)
        scores = {a: np.empty((len(idx), 8), np.float32) for a in ARMS}
        positives = {a: np.empty(len(idx), np.float32) for a in ARMS}
        if uses_cached_controls(cfg):
            with np.load(out/f'control-scores-rank-{rank:02d}.npz', allow_pickle=False) as f:
                np.testing.assert_array_equal(f['positions'], idx)
                np.testing.assert_array_equal(f['source_ids'], panel['source_ids'][idx])
                for a in ('general', 'continued'):
                    scores[a][:], positives[a][:] = f[a], f[a+'_truth']
        if rank == 0:
            run.summary['phase'] = 'scoring_fixed_candidates'
        with torch.no_grad(), feature_precision():
            for start in range(0, len(idx), 256):
                take = idx[start:start+256]
                dest = slice(start, start+len(take))
                b = panel_batch(panel, take, reward.spec, device)
                prepared = reward.prepare_inputs(b)
                c = panel['condition'][take]
                ct = torch.tensor(c, device=device)
                def inputs(feature, delta):
                    extra = correction_features(cfg, panel['visible_a'][take], panel['visible_b'][take], delta)
                    return torch.tensor(feature, device=device), torch.tensor(np.c_[feature, extra], device=device)
                basic, extended = inputs(panel['candidate_truth'][take], panel['truth_deltas'][take])
                for a, model in models.items():
                    positives[a][dest] = model(ct, extended if a == 'residual' else basic).cpu().numpy()
                for j in range(8):
                    d = np.array(deltas[take, j])
                    cc, features = reward.features(b, torch.tensor(d, device=device), prepared=prepared)
                    np.testing.assert_allclose(cc, c, rtol=1e-6, atol=1e-6)
                    if j == 0:
                        np.testing.assert_allclose(features, candidate0[take], rtol=1e-4, atol=1e-3)
                    basic, extended = inputs(features, d)
                    for a, model in models.items():
                        scores[a][dest, j] = model(ct, extended if a == 'residual' else basic).cpu().numpy()
                if rank == 0:
                    run.log({'scoring/events_per_rank': start+len(take)})
        np.savez(out/f'scores-rank-{rank:02d}.npz', positions=idx, source_ids=panel['source_ids'][idx],
                 **scores, **{a+'_truth': value for a, value in positives.items()})
        barrier(world)
        if rank == 0:
            run.summary['phase'] = 'paired_physics_comparison'
            report = finish(cfg, run)
            print('DECISION:', json.dumps(report['decision']), flush=True)
        barrier(world)
        ray.train.report({'completed': 1, 'policy_updates': 0})
    except BaseException:
        if run is not None:
            run.finish(exit_code=1)
            run = None
        raise
    finally:
        if run is not None:
            run.finish()


def main(*, expected_correction=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('config', type=Path)
    p.add_argument('phase', choices=('check', 'verify', 'prepare', 'run'), nargs='?', default='check')
    p.add_argument('--ray-address', default=os.environ.get('RAY_ADDRESS') or 'auto')
    args = p.parse_args()
    cfg = settings(args.config)
    if expected_correction is not None and correction_kind(cfg) != expected_correction:
        raise ValueError('Launcher requires correction='+expected_correction)
    if args.phase == 'check':
        print('CONFIG OK: step500 classifier-only, fixed General + '+correction_kind(cfg)+', matched continuation,16 GPUs,cap30')
        return
    verified = verify(cfg)
    if args.phase == 'verify':
        return
    out = prepare(cfg, verified)
    if args.phase == 'prepare':
        print('PREPARED:', out)
        return
    if (out/'STARTED').exists():
        raise ValueError('Already started; choose a new readable sequence number for a deliberate repeat')
    import ray
    from ray.train import RunConfig, ScalingConfig, FailureConfig
    from ray.train.torch import TorchTrainer
    ray.init(address=args.ray_address, runtime_env={'env_vars': {'PYTHONPATH': os.pathsep.join(
        (str(ROOT), str(ROOT/'scripts'), str(ROOT/'evenet_dgpo')))}})
    try:
        if ray.cluster_resources().get('GPU', 0) < 16:
            raise ValueError('Use an existing 16-GPU Ray allocation')
        (out/'STARTED').write_text('User launched classifier-only continuation and '+correction_kind(cfg)+'.\n')
        TorchTrainer(worker, train_loop_config=cfg, scaling_config=ScalingConfig(num_workers=16, use_gpu=True),
                     run_config=RunConfig(name='general-'+correction_kind(cfg).replace('_','-')+'-correction', storage_path=str(out/'ray_results'),
                                          failure_config=FailureConfig(max_failures=0))).fit()
    finally:
        ray.shutdown()
    print('REPORT:', out/'report.json')


if __name__ == '__main__':
    main()
