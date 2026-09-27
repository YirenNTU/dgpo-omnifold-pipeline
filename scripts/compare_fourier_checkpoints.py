#!/usr/bin/env python3
"""Paired frozen-generator comparison using independent cold H4 audits."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'evenet_dgpo'))
ARMS = ('baseline', 'candidate')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def validate_spec(spec):
    from RL.DGPO_neutrino.omnifold_ztautau.ratio_fit import RatioFitConfig
    if tuple(spec['checkpoints']) != ARMS:
        raise ValueError('Specify baseline and candidate checkpoints in that order')
    for key in ('panel_events', 'generation_batch_size', 'score_batch_size', 'ddim_steps', 'candidates_per_event'):
        if type(spec[key]) is not int or spec[key] < 1:
            raise ValueError(f'Invalid {key}')
    if spec['panel_events'] < 100 or spec['bootstrap_replicates'] < 100:
        raise ValueError('Panel and bootstrap budgets are too small')
    epoch = spec.get('expected_epoch')
    if epoch is not None and (type(epoch) is not int or epoch < 0):
        raise ValueError('expected_epoch must be a nonnegative integer or null')
    if spec.get('checkpoint_selection', 'configured') not in ('configured', 'latest_common'):
        raise ValueError('Unknown checkpoint_selection')
    workers = spec.get('workers', 1)
    if type(workers) is not int or workers < 1 or workers > spec['panel_events']:
        raise ValueError('Invalid workers')
    if workers > 1 and spec['device'] != 'cuda':
        raise ValueError('Multiple workers require CUDA and an existing Ray cluster')
    if spec['sampler'] not in ('legacy', 'stable_v'):
        raise ValueError('Unknown DDIM sampler')
    seeds = spec['classifier_seeds']
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError('Use at least three distinct paired classifier seeds')
    fit = RatioFitConfig(**spec['classifier_fit'])
    fit.validate()
    if workers > 1 and (fit.batch_size % workers or not fit.drop_last_batch):
        raise ValueError('Distributed audits require global batch_size divisible by workers and drop_last_batch: true')
    if fit.min_steps < 1000 or not fit.restore_best or fit.checkpoint_selection_metric != 'loss':
        raise ValueError('Audits need >=1000 updates and validation-BCE selection')
    if fit.validation_interval_steps < 1 or fit.validation_patience_evaluations < 1:
        raise ValueError('Audits require validation plateau detection')
    if fit.require_saturation or fit.ratio_audit_export_dir is not None:
        raise ValueError('Use report-only capped audits; this runner owns its exports')
    if not 0 < spec['material_auc_gap'] < .5:
        raise ValueError('Invalid material AUC-gap margin')
    c = spec['classifier']
    if not (c.get('periodic_pair_features') and c.get('topology_fourier_embedding') and
            c.get('topology_max_harmonic') == 4):
        raise ValueError('The declared evaluator is a common H4 classifier')
    if c.get('topology_direct_logit', False) or c.get('conditional_residual_rank', 0):
        raise ValueError('Use the declared common H4 head, without additional classifier interventions')


def runtime_contract(a, b):
    # Architectures may differ; data, preprocessing and source must not.
    for key in ('data_parquet_dir', 'data_parquet_val_dir'):
        if a['platform'][key] != b['platform'][key]:
            raise ValueError(f'Generator data mismatch: {key}')
    if a['options']['Dataset'] != b['options']['Dataset']:
        raise ValueError('Generator dataset/preprocessing settings differ')
    if a.get('event_info') != b.get('event_info'):
        raise ValueError('Generator event schemas differ')
    for key in ('seed', 'epochs', 'pretrain_model_load_path'):
        if a['options']['Training'][key] != b['options']['Training'][key]:
            raise ValueError(f'Generator training contract differs: {key}')


def load_raw(model, checkpoint):
    source = {k.removeprefix('model.'): v for k, v in checkpoint['state_dict'].items()}
    target = model.state_dict()
    ignored = {'famo', 'Classification', 'Regression', 'Assignment', 'Segmentation',
               'GlobalGeneration', 'ReconGeneration'}
    missing = set(target) - set(source)
    extra = [k for k in source if k not in target and k.split('.')[0] not in ignored]
    if missing or extra:
        raise ValueError(f'Raw checkpoint mismatch: missing={sorted(missing)}, extra={extra}')
    selected = {k: source[k] for k in target}
    if any(not torch.isfinite(v).all() for v in selected.values()):
        raise ValueError('Nonfinite checkpoint weights')
    model.load_state_dict(selected, strict=True)


def checkpoint_provenance(path, expected_epoch=None):
    resolved = Path(path).resolve(strict=True)
    checkpoint = torch.load(resolved, map_location='cpu', weights_only=False)
    if type(checkpoint.get('epoch')) is not int or checkpoint['epoch'] < 0:
        raise ValueError(f'{resolved}: missing/invalid checkpoint epoch')
    if expected_epoch is not None and checkpoint['epoch'] != expected_epoch:
        raise ValueError(f'{resolved}: expected epoch {expected_epoch}, got {checkpoint.get("epoch")}')
    return checkpoint, dict(path=str(resolved), epoch=checkpoint['epoch'],
                            global_step=checkpoint.get('global_step'))


def resolve_checkpoints(spec):
    """Inspect actual metadata, never infer completed epochs from a run budget."""
    inventory = {}
    for arm in ARMS:
        configured = Path(spec['checkpoints'][arm]['checkpoint']).expanduser()
        paths = [configured]
        if spec.get('checkpoint_selection', 'configured') == 'latest_common':
            paths.extend(sorted(configured.parent.glob('*.ckpt')))
        unique = sorted({p.resolve(strict=True) for p in paths if p.is_file()})
        if not unique:
            raise FileNotFoundError(f'{arm}: no checkpoints at {configured}')
        inventory[arm] = []
        for path in unique:
            checkpoint, meta = checkpoint_provenance(path)
            inventory[arm].append(meta)
            del checkpoint
    print(json.dumps({'available_checkpoints':inventory}, indent=2), flush=True)
    expected = spec.get('expected_epoch')
    mode = spec.get('checkpoint_selection', 'configured')
    if mode == 'configured':
        selected = {a:inventory[a][0] for a in ARMS}
        if expected is not None and any(m['epoch'] != expected for m in selected.values()):
            raise ValueError(f'Configured checkpoints do not both match expected_epoch={expected}')
    else:
        common = set.intersection(*[{m['epoch'] for m in inventory[a]} for a in ARMS])
        if expected is not None:
            common &= {expected}
        if not common:
            raise ValueError('No common saved checkpoint epoch' +
                (f' matching expected_epoch={expected}' if expected is not None else '') +
                '. Available checkpoints are listed above. Use exact matched paths, or '
                'checkpoint_selection: configured to compare endpoint quality with recorded epoch differences.')
        epoch = max(common)
        selected = {}
        for arm in ARMS:
            matches = [m for m in inventory[arm] if m['epoch'] == epoch]
            if len(matches) != 1:
                raise ValueError(f'{arm}: ambiguous checkpoint epoch {epoch}; '
                                 'use checkpoint_selection: configured with exact paths')
            selected[arm] = matches[0]
    steps = [selected[a]['global_step'] for a in ARMS]
    epochs = [selected[a]['epoch'] for a in ARMS]
    matched = epochs[0] == epochs[1] and steps[0] is not None and steps[0] == steps[1]
    if mode == 'latest_common' and not matched:
        raise ValueError(f'Checkpoint update budgets differ at epoch {epoch}: {steps}')
    caveat = ('Checkpoint epochs/update counts differ or are unavailable: compare these endpoints\' '
              'generation quality; this does not isolate the effect of Fourier at a matched training budget.')
    return dict(selection=mode, matched_training_budget=matched,
                checkpoints=selected, available=inventory, limitation=None if matched else caveat)


def select_panel(raw, spec):
    import pyarrow.parquet as pq
    from evenet.dataset.preprocess import unflatten_dict
    from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import pack_event_inputs
    directory = Path(spec.get('evaluation_data_dir') or raw['platform']['data_parquet_val_dir'])
    files = sorted(directory.glob('*.parquet'))
    counts = [pq.ParquetFile(p).metadata.num_rows for p in files]
    total = sum(counts)
    if total < spec['panel_events']:
        raise ValueError(f'Only {total} evaluation rows, need {spec["panel_events"]}')
    selected = np.sort(np.random.default_rng(spec['panel_seed']).choice(total, spec['panel_events'], replace=False))
    metadata = json.loads((Path(raw['platform']['data_parquet_dir']) / 'shape_metadata.json').read_text())
    pieces, truth, manifest, packing = [], [], [], None
    offset = 0
    for path, count in zip(files, counts):
        local = selected[(selected >= offset) & (selected < offset + count)] - offset
        if len(local):
            row = 0
            for record in pq.ParquetFile(path).iter_batches(batch_size=4096):
                ids = local[(local >= row) & (local < row + record.num_rows)] - row
                if len(ids):
                    sub = record.take(ids)
                    flat = {name: sub.column(i).to_numpy(zero_copy_only=False)
                            for i, name in enumerate(sub.schema.names)}
                    arrays = unflatten_dict(flat, metadata,
                        drop_column_prefix=['EXTRA/', 'regression-', 'assignments-', 'segmentation-'])
                    batch = {k: torch.as_tensor(v.copy()) for k, v in arrays.items()}
                    z = batch['x_invisible']
                    if z.shape[1:] != (2, 2) or not batch['x_invisible_mask'].bool().all():
                        raise ValueError('This evaluator requires two valid tau targets, each delta theta/phi')
                    packed, packing = pack_event_inputs(batch, packing, include_pairwise_context=True)
                    # Padding does not define event identity; require finite packed inputs.
                    if not torch.isfinite(packed).all() or not torch.isfinite(z).all():
                        raise ValueError('Nonfinite evaluation data; refusing to silently drop rows')
                    pieces.append(packed); truth.append(z.reshape(len(z), 4))
                    manifest.append(dict(file=str(path), rows=(row + ids).tolist()))
                row += record.num_rows
        offset += count
    return torch.cat(pieces), torch.cat(truth), packing, manifest, str(directory)


def identity_splits(condition, seed):
    from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import _identity_crossfit_splits
    folds = _identity_crossfit_splits(condition, folds=5, seed=seed)
    holdouts = [pair[1].cpu() for pair in folds]
    return dict(fit=torch.cat(holdouts[2:]), early_stop=holdouts[1], test=holdouts[0])


def sample_policy(model, condition, packing, noise, spec, device, emit):
    from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import unpack_event_inputs
    from RL.DGPO_neutrino.sampling import generate_neutrino_candidates
    from diagnose_h4_ddim_coverage import make_replay_sampler
    if model.invisible_input_dim != noise.shape[-1]:
        raise ValueError('Initial noise/normalizer feature dimension mismatch')
    generated = []
    model.eval().requires_grad_(False)
    for start in range(0, len(condition), spec['generation_batch_size']):
        c = condition[start:start + spec['generation_batch_size']].to(device)
        batch = unpack_event_inputs(c, packing)
        # Only shape/masks enter inference, never held-out target values.
        batch['x_invisible'] = torch.zeros(len(c), 2, 2, device=device)
        batch['x_invisible_mask'] = torch.ones(len(c), 2, dtype=torch.bool, device=device)
        bank = noise[start:start + len(c)].permute(1, 0, 2, 3).to(device)
        sampler = make_replay_sampler(bank, spec['sampler'])
        out = generate_neutrino_candidates(model, batch, sampler,
            K=spec['candidates_per_event'], num_ddim_steps=spec['ddim_steps'],
            device=device, parallel_chains=1)
        if sampler.draws_used != spec['candidates_per_event'] or out.shape != (spec['candidates_per_event'], len(c), 2, 2):
            raise ValueError('Sampling/noise replay contract changed')
        if not torch.isfinite(out).all():
            raise ValueError('Nonfinite generated candidates; no clipping or rejection sampling allowed')
        generated.append(out.permute(1, 0, 2, 3).reshape(len(c), spec['candidates_per_event'], 4).cpu())
        emit(dict(events_sampled=start+len(c)))
    return torch.cat(generated)


def classifier_metrics(truth, generated):
    t, g = np.asarray(truth), np.asarray(generated)
    if t.shape != g.shape or t.ndim != 1 or not np.isfinite(np.r_[t, g]).all():
        raise ValueError('Invalid paired classifier scores')
    auc = roc_auc_score(np.r_[np.ones(len(t)), np.zeros(len(g))], np.r_[t, g])
    return dict(auc=float(auc), auc_gap=float(abs(auc-.5)),
                bce=float(.5*(np.logaddexp(0, -t).mean()+np.logaddexp(0, g).mean())))


def paired_auc_comparison(scores, identities, replicates, seed, margin, ready):
    # Keep all duplicates of an event together; same indices for every model/arm.
    _, inverse = np.unique(np.asarray(identities), axis=0, return_inverse=True)
    groups = [np.flatnonzero(inverse == j) for j in range(inverse.max()+1)]
    per_seed = [classifier_metrics(s['candidate'][0], s['candidate'][1])['auc_gap'] -
                classifier_metrics(s['baseline'][0], s['baseline'][1])['auc_gap'] for s in scores]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(replicates):
        ids = np.concatenate([groups[j] for j in rng.integers(len(groups), size=len(groups))])
        draws.append(np.mean([classifier_metrics(s['candidate'][0][ids], s['candidate'][1][ids])['auc_gap'] -
                              classifier_metrics(s['baseline'][0][ids], s['baseline'][1][ids])['auc_gap'] for s in scores]))
    lo, hi = np.quantile(draws, [.025, .975])
    decision = 'inconclusive'
    if not ready:
        decision = 'inconclusive_undertrained_or_unsaturated_audit'
    elif hi < -margin and max(per_seed) < 0:
        decision = 'candidate_improves_h4_gap'
    elif lo > margin and min(per_seed) > 0:
        decision = 'candidate_worsens_h4_gap'
    elif lo >= -margin and hi <= margin:
        decision = 'no_material_gap_difference_on_this_panel'
    return dict(candidate_minus_baseline_mean_gap=float(np.mean(per_seed)),
        per_classifier_seed_delta=per_seed, paired_event_ci95=[float(lo), float(hi)],
        material_margin=margin, all_audits_ready=ready, decision=decision,
        uncertainty_scope='Event-cluster bootstrap conditional on fitted judges and fixed sampling noise; '
                          'classifier seed spread is reported separately, not a generator-training replication.')


def physics_metrics(condition, truth, generated, packing):
    from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import unpack_event_inputs
    from diagnose_h4_topology_resolution import reconstruct, w1
    fields = unpack_event_inputs(condition.cpu(), packing)
    target = reconstruct(fields, truth.cpu())
    candidates = [reconstruct(fields, generated[:, k].cpu()) for k in range(generated.shape[1])]
    # Prespecified dimensionless visible-angle features times joint tau observables.
    condition_features = [np.ones(len(truth))]
    for leg in ('a', 'b'):
        p = np.stack([fields[f'lead_{leg}_visible_{axis}'].numpy().reshape(-1) for axis in ('px','py','pz')], -1)
        theta, phi = np.arctan2(np.hypot(p[:,0],p[:,1]), p[:,2]), np.arctan2(p[:,1],p[:,0])
        for angle in (theta, phi):
            for k in (1,2,4,8):
                condition_features.extend([np.sin(k*angle), np.cos(k*angle)])
    c = np.stack(condition_features, -1)
    moments = c[:, :, None] * (np.mean([x['topology'] for x in candidates], 0) - target['topology'])[:, None, :]
    report = dict(conditional_joint_moment_rmse=float(np.sqrt(np.mean(moments.mean(0)**2))),
        angular_w1_radians={}, invalid_direction_inputs={key:sum(x['invalid'][key] for x in candidates) for key in target['invalid']})
    for key in ('acoplanarity', 'acollinearity'):
        g = np.stack([x[key] for x in candidates], 1).reshape(-1)
        report['angular_w1_radians'][key] = w1(target[key], g, np.ones(len(g)))
    return report


def write_review(output, report):
    comp = report['comparison']
    lines = ['# Paired Fourier checkpoint evaluation', '',
        f"Decision: **{comp['decision']}**", '',
        'Primary endpoint: held-out cold H4 AUC gap, averaged over paired classifier seeds.',
        f"Candidate minus baseline: {comp['candidate_minus_baseline_mean_gap']:.6f}; "
        f"paired event 95% interval: {comp['paired_event_ci95']}. Negative favors candidate.", '',
        '| Checkpoint | Epoch | Updates | Resolved file |',
        '| --- | ---: | ---: | --- |',
        *[f"| {a} | {m['epoch']} | {m['global_step']} | {m['path']} |" for a,m in report['checkpoints'].items()], '',
        '| Audit | Updates | Best step | Plateau | Test AUC | Test gap | Test BCE |',
        '| --- | ---: | ---: | --- | ---: | ---: | ---: |']
    for key, a in report['audits'].items():
        f, m = a['fit'], a['matched_test']
        lines.append(f"| {key} | {f['steps_completed']} | {f['best_step']} | {f['saturated']} | "
                     f"{m['auc']:.6f} | {m['auc_gap']:.6f} | {m['bce']:.6f} |")
    lines += ['', 'Secondary diagnostics (same held-out events; all unselected draws):', '',
        '| Arm | Conditional joint moment RMSE | Acoplanarity W1 | Acollinearity W1 |',
        '| --- | ---: | ---: | ---: |']
    for arm, m in report['physics_secondary'].items():
        w = m['angular_w1_radians']
        lines.append(f"| {arm} | {m['conditional_joint_moment_rmse']:.6g} | "
                     f"{w['acoplanarity']:.6g} | {w['acollinearity']:.6g} |")
    lines += ['', 'Limitations:', '', *['- '+s for s in report['limitations']], '',
        'See report.json for the cross-judge score matrix, seed differences and full fit histories.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')


def prepare(spec, output, selection):
    from evenet.control.global_config import Config
    raw = {a: yaml.safe_load(Path(spec['checkpoints'][a]['runtime_config']).read_text()) for a in ARMS}
    runtime_contract(raw['baseline'], raw['candidate'])
    configs = {}
    provenance = selection['checkpoints']
    for arm in ARMS:
        configs[arm] = Config(); configs[arm].load_yaml(spec['checkpoints'][arm]['runtime_config'])
        (output / f'{arm}_runtime.yaml').write_text(yaml.safe_dump(raw[arm], sort_keys=False))
    if raw['baseline']['network']['Body']['PET'].get('visible_angular_fourier', {}).get('enabled', False):
        raise ValueError('Use the no-Fourier baseline runtime for the common classifier foundation')
    if not Path(spec['classifier_backbone']).is_file():
        raise FileNotFoundError(spec['classifier_backbone'])
    c, truth, packing, manifest, data_dir = select_panel(raw['baseline'], spec)
    splits = identity_splits(c, spec['split_seed'])
    if min(len(x) for x in splits.values()) < 10 or len(splits['fit']) < spec['classifier_fit']['batch_size']:
        raise ValueError('An identity fold is too small for the declared fit/evaluation budget')
    norm_path = configs['baseline'].options.Dataset.normalization_file
    normalization = torch.load(norm_path, map_location='cpu', weights_only=False)
    noise = torch.randn(len(c), spec['candidates_per_event'], 2, len(normalization['invisible_mean']['Source']),
                        generator=torch.Generator().manual_seed(spec['noise_seed']))
    torch.save(dict(condition=c, truth=truth, packing_spec=packing.to_dict(), splits=splits,
                    initial_noise=noise), output/'panel.pt')
    write_json(output/'manifest.json', dict(spec=spec, checkpoints=provenance,
        checkpoint_selection=selection, rows=manifest,
        evaluation_data_dir=data_dir, generator_weights_updated=False,
        selection_caveat='The default panel is generator validation data, not an untouched test population.'))


def distributed_context():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def barrier():
    if distributed_context()[1] > 1:
        torch.distributed.barrier()


def event_positions(events, rank, world):
    if world < 1 or not 0 <= rank < world or events < world:
        raise ValueError('Invalid generation shard')
    return torch.arange(rank, events, world)


def merge_generation_shards(parts, events, candidates):
    merged = torch.empty(events, candidates, 4)
    seen = torch.zeros(events, dtype=torch.bool)
    for part in parts:
        positions, values = part['positions'], part['generated']
        if positions.dtype != torch.int64 or positions.ndim != 1 or (positions < 0).any() or (positions >= events).any():
            raise ValueError('Invalid generation positions')
        if len(positions.unique()) != len(positions) or seen[positions].any():
            raise ValueError('Duplicate generation events')
        if values.shape != (len(positions), candidates, 4) or not torch.isfinite(values).all():
            raise ValueError('Invalid generation values')
        merged[positions] = values
        seen[positions] = True
    if not seen.all():
        raise ValueError('Missing generation events')
    return merged


def run(spec, output, emit, device=None):
    from evenet.control.global_config import Config
    from evenet.network.evenet_model import build_evenet_model_from_training_config
    from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import (
        EvenetAdapterModelBuilder, EventPackingSpec, fit_independent_evenet_audit, _score_population)
    from RL.DGPO_neutrino.omnifold_ztautau.ratio_fit import RatioFitConfig
    rank, world = distributed_context()
    if world != spec.get('workers', 1):
        raise ValueError(f"Expected {spec.get('workers', 1)} workers, got {world}")
    device = torch.device(spec['device']) if device is None else device
    configs = {}
    for arm in ARMS:
        configs[arm] = Config(); configs[arm].load_yaml(output/f'{arm}_runtime.yaml')
    manifest = json.loads((output/'manifest.json').read_text())
    provenance = manifest['checkpoints']
    panel = torch.load(output/'panel.pt', map_location='cpu', weights_only=True)
    c, truth, noise, splits = (panel[k] for k in ('condition','truth','initial_noise','splits'))
    packing = EventPackingSpec.from_dict(panel['packing_spec'])
    normalization = torch.load(configs['baseline'].options.Dataset.normalization_file,
                               map_location='cpu', weights_only=False)
    positions = event_positions(len(c), rank, world)
    generated = {}
    for arm in ARMS:
        torch.manual_seed(42)
        model = build_evenet_model_from_training_config(configs[arm], normalization, device).to(device)
        ckpt = torch.load(provenance[arm]['path'], map_location='cpu', weights_only=False)
        if any(ckpt.get(k) != provenance[arm][k] for k in ('epoch', 'global_step')):
            raise ValueError('Checkpoint metadata changed after preflight')
        load_raw(model, ckpt); del ckpt
        local = sample_policy(model, c[positions], packing, noise[positions], spec, device,
            lambda row: emit({f'generation/{arm}/rank0_{k}':v for k,v in row.items()}))
        shard_path = output/f'{arm}_rank{rank:03d}.pt'
        torch.save(dict(positions=positions, generated=local), shard_path)
        del local
        barrier()
        if rank == 0:
            parts = (torch.load(output/f'{arm}_rank{i:03d}.pt', map_location='cpu', weights_only=True)
                     for i in range(world))
            merged = merge_generation_shards(parts, len(c), spec['candidates_per_event'])
            torch.save(merged, output/f'{arm}_candidates.pt')
            emit({f'generation/{arm}/events_completed':len(c)})
        barrier()
        generated[arm] = torch.load(output/f'{arm}_candidates.pt', map_location='cpu', weights_only=True)
        del model; gc.collect()
        if device.type == 'cuda': torch.cuda.empty_cache()
    # A common no-Fourier classifier foundation, independent of either endpoint.
    torch.manual_seed(42)
    builder = EvenetAdapterModelBuilder(config=configs['baseline'], normalization_dict=normalization,
        checkpoint_path=spec['classifier_backbone'], device=device, **spec['classifier'])
    cg, tg = c.to(device), truth.to(device)
    test = splits['test']
    summary, paired_scores, all_ready = {}, [], True
    report = dict(checkpoints=provenance, primary_endpoint='cold_H4_test_abs_auc_minus_0.5',
                  checkpoint_selection=manifest['checkpoint_selection'],
                  workers=world, global_classifier_batch_size=spec['classifier_fit']['batch_size'],
                  split_events={k:len(v) for k,v in splits.items()}, audits=summary)
    for seed in spec['classifier_seeds']:
        seed_scores = {}
        for arm in ARMS:
            # Pair dropout streams across arms, with distinct streams across ranks.
            torch.manual_seed(int(seed) + 1000003*rank)
            holder = []
            def factory():
                model = builder.make_classifier(packing, reset=True)
                holder.append(model)
                return model
            prefix = f'audit/{arm}/seed{seed}'
            result = fit_independent_evenet_audit(model_factory=factory,
                data_condition=cg, data_sample=tg, gen_condition=cg,
                gen_sample=generated[arm][:,0].to(device), gen_weight=torch.ones(len(c), device=device),
                fit_config=RatioFitConfig(**spec['classifier_fit']), seed=seed,
                identity_split_seed=spec['split_seed'], reuse_early_stop_for_audit=False,
                progress_callback=lambda row: emit({f'{prefix}/{k}':v for k,v in row.items()}))
            model = holder.pop().eval()
            ready = bool(result.fit_diagnostics.saturated and
                         (result.fit_diagnostics.steps_completed or 0) >= spec['classifier_fit']['min_steps'])
            if result.warm_started or (result.fit_events, result.early_stop_events, result.audit_events) != (
                    len(splits['fit']), len(splits['early_stop']), len(splits['test'])):
                raise ValueError('Cold/disjoint audit contract changed')
            all_ready &= ready
            tlog = _score_population(model, cg[test], tg[test], spec['score_batch_size']).detach().cpu().numpy().reshape(-1)
            cross = {}
            for tested in ARMS:
                glog = _score_population(model, cg[test], generated[tested][test,0].to(device), spec['score_batch_size']).detach().cpu().numpy().reshape(-1)
                cross[tested] = classifier_metrics(tlog, glog)
                if tested == arm:
                    seed_scores[arm] = (tlog, glog)
                if rank == 0:
                    np.savez_compressed(output/f'judge_{arm}_seed{seed}_on_{tested}.npz',
                                        test_rows=test.numpy(), truth_logits=tlog, generated_logits=glog)
            if not np.isclose(cross[arm]['auc'], result.auc, atol=1e-6, rtol=0):
                raise ValueError('Reported audit and saved test scores disagree; check split alignment')
            if rank == 0:
                torch.save(model.state_dict(), output/f'judge_{arm}_seed{seed}.pt')
            key = f'{arm}_seed{seed}'
            summary[key] = dict(fit=asdict(result.fit_diagnostics), ready=ready,
                matched_test=cross[arm], cross_scores=cross, warm_started=result.warm_started)
            emit({f'{prefix}/test/{k}':v for k,v in cross[arm].items()})
            if rank == 0:
                write_json(output/'report.partial.json', report)
            del model; gc.collect()
            if device.type == 'cuda': torch.cuda.empty_cache()
        paired_scores.append(seed_scores)
    if rank == 0:
        report['comparison'] = paired_auc_comparison(paired_scores, c[test].numpy(),
            spec['bootstrap_replicates'], spec['bootstrap_seed'], spec['material_auc_gap'], all_ready)
        report['physics_secondary'] = {a:physics_metrics(c[test], truth[test], generated[a][test], packing) for a in ARMS}
        report['limitations'] = ['AUC depends on the evaluator family; plateau is a training check, not proof of closure.',
            'Both generator checkpoints come from one training seed.',
            'Default data were previously used for generator validation/model selection.',
            'Three classifier seeds share one generation noise panel; bootstrap is conditional on fitted judges.']
        if manifest['checkpoint_selection']['limitation']:
            report['limitations'].insert(0, manifest['checkpoint_selection']['limitation'])
        write_json(output/'report.json', report)
        write_review(output, report)
        (output/'COMPLETE').write_text('fourier-checkpoint-comparison-v1\n')
        emit({f'comparison/{k}':v for k,v in report['comparison'].items() if isinstance(v,(int,float,str,bool))})
    barrier()
    return report if rank == 0 else None


def execute(spec, output, device=None):
    rank, _ = distributed_context()
    run_wandb = None
    success = False
    try:
        if rank == 0 and spec['wandb']['enabled']:
            import wandb
            w = spec['wandb']
            manifest = json.loads((output/'manifest.json').read_text())
            run_wandb = wandb.init(entity=w['entity'], project=w['project'], group=w['group'], name=w['name'],
                id=wandb.util.generate_id(), resume='never',
                config={**spec, 'resolved_checkpoints':manifest['checkpoints'],
                        'resolved_selection':manifest['checkpoint_selection'], 'attempt_output_dir':str(output)},
                tags=['FourierComparison','ColdH4','PairedEvents','NoPolicyUpdate'])
        progress = output/'progress.jsonl'
        def emit(row):
            if rank != 0:
                return
            print(json.dumps(row, allow_nan=False), flush=True)
            with progress.open('a') as f: f.write(json.dumps(row, allow_nan=False)+'\n')
            if run_wandb: run_wandb.log(row)
        report = run(spec, output, emit, device)
        if run_wandb: run_wandb.summary['comparison'] = report['comparison']
        if rank == 0: print(json.dumps(report['comparison'], indent=2))
        success = True
    finally:
        if run_wandb: run_wandb.finish(exit_code=0 if success else 1)


def worker(config):
    import ray.train
    from ray.train.torch import get_device
    device = get_device()
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    # The production ratio fitter synchronizes gradients itself; do not wrap DDP.
    execute(config['spec'], Path(config['output']), device)
    ray.train.report({'complete':1})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT/'config/compare_fourier_checkpoints.yaml')
    p.add_argument('--check-only', action='store_true', help='Validate YAML only; no data/cluster access')
    p.add_argument('--inspect-checkpoints', action='store_true', help='Read checkpoint metadata only; no Ray, W&B or fits')
    p.add_argument('--ray-address', default=os.environ.get('RAY_ADDRESS') or 'auto')
    args = p.parse_args(argv)
    spec = yaml.safe_load(args.config.read_text()); validate_spec(spec)
    if args.check_only:
        print(yaml.safe_dump(spec, sort_keys=False)); return
    if int(os.environ.get('SLURM_PROCID', '0')) != 0:
        p.error('Launch this driver once inside the allocation, not once per GPU')
    selection = resolve_checkpoints(spec)
    print(json.dumps({'selected_checkpoints':selection['checkpoints'],
                      'matched_training_budget':selection['matched_training_budget'],
                      'limitation':selection['limitation']}, indent=2), flush=True)
    if args.inspect_checkpoints:
        return
    workers = spec.get('workers', 1)
    if workers > 1:
        import ray
        from ray.train import RunConfig, ScalingConfig, FailureConfig
        from ray.train.torch import TorchTrainer
        if args.ray_address == 'local':
            raise ValueError('Use an existing allocated Ray cluster; local fallback is disabled')
        ray.init(address=args.ray_address, runtime_env={'env_vars':{
            'PYTHONPATH':os.pathsep.join([str(ROOT/'evenet_dgpo'), str(ROOT/'scripts'), os.environ.get('PYTHONPATH','')]),
            'OMP_NUM_THREADS':'2', 'OPENBLAS_NUM_THREADS':'1'}})
        available = ray.available_resources()
        if available.get('GPU', 0) < workers or available.get('CPU', 0) < 2*workers:
            raise RuntimeError(f'Requires {workers} free GPUs and {2*workers} CPUs in the existing Ray cluster; '
                               f'available={available}. No job was submitted.')
    attempt = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    epochs = '_'.join(f'{a}_epoch{selection["checkpoints"][a]["epoch"]}' for a in ARMS)
    output = Path(spec['output_dir']).expanduser().resolve()/f'{epochs}_{attempt}'
    output.mkdir(parents=True, exist_ok=False)
    print(f'Output: {output}', flush=True)
    write_json(output/'spec.json', spec)
    try:
        prepare(spec, output, selection)
        if workers == 1:
            execute(spec, output)
        else:
            TorchTrainer(train_loop_per_worker=worker,
                train_loop_config={'spec':spec, 'output':str(output)},
                scaling_config=ScalingConfig(num_workers=workers, use_gpu=True,
                                             resources_per_worker={'CPU':2, 'GPU':1}),
                run_config=RunConfig(name=f'fourier-comparison-{attempt}',
                    storage_path=str(output/'ray_results'), failure_config=FailureConfig(max_failures=0))).fit()
        print(f'Full report: {output / "REPORT.md"}', flush=True)
    except Exception as exc:
        write_json(output/'FAILED.json', dict(error=str(exc)))
        raise


if __name__ == '__main__': main()
