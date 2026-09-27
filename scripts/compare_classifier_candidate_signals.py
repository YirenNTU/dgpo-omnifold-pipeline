#!/usr/bin/env python3
"""Frozen old/pair classifiers on the same held-out K8 candidate panel.

Score comparison only: no fit, policy update, calibration, or validated
cross-generator density-ratio interpretation.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

from diagnose_h4_classifier_calibration import build_classifier, original_shard, score, verify_logits
from diagnose_h4_spike_coverage import ROOT, load_artifact, write_json


def aligned_rows(bundle, panel):
    """Require exact held-out identity, context, truth and representation."""
    if bundle['packing_spec'] != panel['packing_spec']:
        raise ValueError('Packing specifications differ')
    splits = bundle['split_indices']
    test = splits['test'].tolist()
    if len(set(test)) != len(test):
        raise ValueError('Duplicate test identities')
    # pool_rows are local positions, not stable identities across Ray exports.
    def row_keys(values):
        return [row.tobytes() for row in values.contiguous().numpy()]
    saved_keys = row_keys(bundle['test_condition'])
    wanted = row_keys(panel['condition'])
    if len(set(saved_keys)) != len(saved_keys) or len(set(wanted)) != len(wanted):
        raise ValueError('Ambiguous duplicate event contexts')
    mapping = {identity: row for row, identity in enumerate(saved_keys)}
    if any(i not in mapping for i in wanted):
        raise ValueError('Panel is not entirely held out by this classifier')
    if set(test) & (set(splits['fit'].tolist()) | set(splits['early_stop'].tolist())):
        raise ValueError('Panel overlaps classifier training or selection')
    ids = torch.tensor([mapping[i] for i in wanted], dtype=torch.long)
    for saved, key in [('test_truth', 'truth'), ('test_condition', 'condition')]:
        if not torch.equal(bundle[saved][ids], panel[key]):
            raise ValueError(f'Exact panel {key} mismatch')
    return ids


def summarize(x):
    x = np.asarray(x, dtype=float)
    return dict(mean=float(x.mean()), q10=float(np.quantile(x, .1)),
                median=float(np.median(x)), q90=float(np.quantile(x, .9)))


def signal_report(old, pair):
    if old.shape != pair.shape or old.ndim != 2 or old.shape[1] < 2:
        raise ValueError('Expected aligned N by K scores, K >= 2')
    if not np.isfinite(old).all() or not np.isfinite(pair).all():
        raise ValueError('Nonfinite candidate scores')
    arms = {}
    for name, values in [('old', old), ('pair', pair)]:
        centered = values - values.mean(1, keepdims=True)
        weights = np.exp(values - values.max(1, keepdims=True))
        weights /= weights.sum(1, keepdims=True)
        arms[name] = dict(within_event_score_sd=summarize(values.std(1)),
            within_event_score_range=summarize(np.ptp(values, axis=1)),
            softmax_score_ess_fraction=summarize(1 / (values.shape[1] * (weights**2).sum(1))),
            max_candidate_score_mass=summarize(weights.max(1)),
            centered_score_rms=float(np.sqrt(np.mean(centered**2))))
    a, b = [v - v.mean(1, keepdims=True) for v in (old, pair)]
    norm = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    valid = norm > 1e-12
    ranks = [rankdata(v, axis=1) for v in (old, pair)]
    ranks = [v - v.mean(1, keepdims=True) for v in ranks]
    rnorm = np.linalg.norm(ranks[0], axis=1) * np.linalg.norm(ranks[1], axis=1)
    rv = rnorm > 0
    i, j = np.triu_indices(old.shape[1], 1)
    da, db = old[:, i] - old[:, j], pair[:, i] - pair[:, j]
    distinct = (da != 0) & (db != 0)
    unique = ((old == old.max(1, keepdims=True)).sum(1) == 1) & ((pair == pair.max(1, keepdims=True)).sum(1) == 1)
    return dict(events=len(old), K=old.shape[1], arms=arms,
        centered_score_cosine=summarize((a[valid]*b[valid]).sum(1)/norm[valid]) if valid.any() else None,
        centered_cosine_valid_events=int(valid.sum()),
        within_event_spearman=summarize((ranks[0][rv]*ranks[1][rv]).sum(1)/rnorm[rv]) if rv.any() else None,
        spearman_valid_events=int(rv.sum()),
        pairwise_order_agreement=float((np.sign(da[distinct]) == np.sign(db[distinct])).mean()) if distinct.any() else None,
        non_tied_candidate_pairs=int(distinct.sum()),
        unique_winner_agreement=float((old.argmax(1)[unique] == pair.argmax(1)[unique]).mean()) if unique.any() else None,
        unique_winner_events=int(unique.sum()),
        limitations=['Descriptive frozen scores on one shared step1110 candidate panel.',
            'Old classifier was trained against pretrain; its scores here are not validated step1110 density ratios.',
            'Softmax ESS is a candidate-score concentration diagnostic, not production DGPO gradient ESS.',
            'Agreement does not establish correct ordering, gradient direction, or policy closure.'])


def worker(cfg):
    import ray.train
    import ray.train.torch
    import yaml
    from evenet.control.global_config import global_config
    rank = ray.train.get_context().get_world_rank()
    world = ray.train.get_context().get_world_size()
    device = ray.train.torch.get_device()
    global_config.load_yaml(cfg['runtime'])
    raw = yaml.safe_load(Path(cfg['runtime']).read_text())
    torch.set_float32_matmul_precision(str(raw['dgpo'].get('float32_matmul_precision', 'medium')))
    b = load_artifact(Path(cfg['artifact']))
    panel = torch.load(Path(cfg['panel'])/'panel.pt', map_location='cpu', weights_only=True)
    aligned_rows(b, panel)
    model = build_classifier(raw, b['model_state'], b['packing_spec'], device)
    start, stop = original_shard(len(b['test_truth']), rank, world)
    batch = int(b['fit_config']['validation_batch_size'])
    replay = {}
    for side, field in [('truth', 'test_truth'), ('gen', 'test_generated')]:
        observed = score(model, b['test_condition'][start:stop], b[field][start:stop].reshape(-1, 4), device, batch)
        replay[side] = verify_logits(observed, b[side+'_logits'].reshape(-1)[start:stop])
    positions = torch.arange(rank, len(panel['truth']), world)
    candidates = torch.load(Path(cfg['panel'])/'candidates.pt', map_location='cpu', weights_only=True)['generated'][positions, :8]
    if candidates.shape != (len(positions), 8, 4):
        raise ValueError('Candidate panel shape differs')
    c = panel['condition'][positions].repeat_interleave(8, dim=0)
    logits = score(model, c, candidates.reshape(-1, 4), device, 128).reshape(-1, 8)
    torch.save(dict(positions=positions, scores=logits, replay_max_error=replay), Path(cfg['output'])/f'rank-{rank:03d}.pt')
    ray.train.report({'events': len(positions)})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old', type=Path, required=True)
    p.add_argument('--pair', type=Path, required=True)
    p.add_argument('--panel', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    import yaml
    import ray
    from ray.train import RunConfig, ScalingConfig, FailureConfig
    from ray.train.torch import TorchTrainer
    from train_neutrino_backend import read_yaml, deep_update, absolutize_default_paths
    if not (args.panel/'COMPLETE').is_file():
        p.error('Coverage panel is incomplete')
    panel = torch.load(args.panel/'panel.pt', map_location='cpu', weights_only=True)
    # Select using identity membership only, never classifier scores.
    bundles = [load_artifact(source) for source in (args.old, args.pair)]
    panel_keys = [row.tobytes() for row in panel['condition'].contiguous().numpy()]
    common = set(panel_keys)
    for b in bundles:
        common &= {row.tobytes() for row in b['test_condition'].contiguous().numpy()}
    positions = torch.tensor([i for i, identity in enumerate(panel_keys) if identity in common], dtype=torch.long)
    if len(positions) < 32:
        p.error(f'Only {len(positions)} shared held-out events; insufficient common panel')
    original_events = len(panel['truth'])
    panel = {key: value[positions] if isinstance(value, torch.Tensor) and value.ndim > 0 and len(value) == original_events else value for key, value in panel.items()}
    for b in bundles:
        aligned_rows(b, panel)
    args.output.mkdir(parents=True, exist_ok=False)
    shared_panel = args.output/'shared_panel'
    shared_panel.mkdir()
    torch.save(panel, shared_panel/'panel.pt')
    candidates = torch.load(args.panel/'candidates.pt', map_location='cpu', weights_only=True)
    torch.save({'generated': candidates['generated'][positions]}, shared_panel/'candidates.pt')
    write_json(args.output/'panel_selection.json', dict(original_events=original_events, shared_heldout_events=len(positions),
        source_panel_positions=positions.tolist(), selection='Intersection of saved held-out identities only; exact context/truth check required.'))
    ray.init(address='auto', runtime_env={'env_vars': {'PYTHONPATH': os.pathsep.join([str(ROOT/'evenet_dgpo'), str(ROOT/'scripts'), str(ROOT)])}})
    scores, errors = {}, {}
    for name, source in [('old', args.old), ('pair', args.pair)]:
        out = args.output/name
        out.mkdir()
        raw = deep_update(read_yaml(ROOT/'config/train_diffusion_nersc.yaml'), read_yaml(source.parent/'resolved_ratio_experiment.yaml'))
        raw = absolutize_default_paths(raw, ROOT/'config')
        raw.setdefault('compat', {}).update(backend='dgpo-evenet', repo_root=str(ROOT))
        raw.setdefault('rl', {})['enabled'] = True
        runtime = out/'runtime.yaml'
        runtime.write_text(yaml.safe_dump(raw, sort_keys=False))
        cfg = dict(artifact=str(source.resolve()), panel=str(shared_panel.resolve()), runtime=str(runtime.resolve()), output=str(out.resolve()))
        TorchTrainer(train_loop_per_worker=worker, train_loop_config=cfg,
            scaling_config=ScalingConfig(num_workers=16, use_gpu=True),
            run_config=RunConfig(name=name, storage_path=str((out/'ray_results').resolve()), failure_config=FailureConfig(max_failures=0))).fit()
        merged = torch.empty(len(panel['truth']), 8)
        seen, errors[name] = [], []
        for rank in range(16):
            part = torch.load(out/f'rank-{rank:03d}.pt', map_location='cpu', weights_only=True)
            merged[part['positions']] = part['scores']
            seen.extend(part['positions'].tolist())
            errors[name].append(part['replay_max_error'])
        if sorted(seen) != list(range(len(panel['truth']))):
            raise ValueError('Missing or duplicate scored events')
        scores[name] = merged
    report = signal_report(scores['old'].numpy(), scores['pair'].numpy())
    report['original_logit_replay_errors'] = errors
    report['sources'] = {name: str(getattr(args, name)) for name in ('old', 'pair', 'panel')}
    write_json(args.output/'report.json', report)
    torch.save(scores, args.output/'scores.pt')
    (args.output/'COMPLETE').write_text('frozen-classifier-candidate-signals-v1\n')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
