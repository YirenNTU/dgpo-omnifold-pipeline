#!/usr/bin/env python3
"""Run A/B/C sequentially on an EXISTING 16-GPU Ray allocation; never submit jobs.

--prepare-only validates and pins files, without starting Ray or W&B.
The default source is the exact late step1130/reward7 source of 6704e92e.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import uuid

import numpy as np
import yaml

from diagnose_dgpo_checkpoint_transfer import build_probe_config, source_metadata, validate_probe_runtime
from train_neutrino_backend import REPO_ROOT, EVENET_DGPO_ROOT, command_for_backend, read_yaml

SOURCE = Path('/pscratch/sd/y/yiren/Ztautau/dgpo_checkpoint_transfer/probe-20260928T163114-ph2iomf6')
CONFIG = REPO_ROOT / 'config/dgpo_conditioning_late_probe.yaml'
OUTPUT = Path('/pscratch/sd/y/yiren/Ztautau/dgpo_conditioning_late_50')
ARMS = {'A': ('Original conditioning', None), 'B': ('Individual angle residual', 'individual'),
        'C': ('Relative angle residual', 'relative')}
STEPS = [0, 1, 5, 20, 35, 50]


def build_arm_config(source_runtime, *, pinned, output, root, metadata, arm, run_id):
    cfg = build_probe_config(source_runtime, CONFIG, pinned=pinned, output=output, metadata=metadata)
    if cfg['dgpo'].get('best_source_start_new_experiment') or cfg['dgpo'].get('pinned_classifier_restart'):
        raise ValueError('Conditioning test cannot reset trained optimizer/reference state')
    label, basis = ARMS[arm]
    visible = cfg['network']['VisibleConditioning']
    if (not visible.get('diffusion_enabled') or visible.get('diffusion_token_readout', {}).get('enabled')
            or visible.get('diffusion_reward_probe', {}).get('enabled')):
        raise ValueError('Source must be the existing global-FiLM policy without a trained probe/token readout')
    if basis is not None:
        visible['diffusion_reward_probe'] = {'enabled': True, 'basis': basis, 'width': 128, 'seed': 20260930}
    probe = cfg['dgpo']['checkpoint_transfer']
    probe.update(relative_steps=STEPS, update_cache_directory=str(root / 'update_cache'),
                 update_cache_mode='write' if arm == 'A' else 'read')
    if arm != 'A':
        probe['paired_panel_directory'] = str(root / 'A' / 'measurements')
    cfg['dgpo']['gradient_transfer_trace']['update_end_steps'] = [metadata['global_step'] + x for x in STEPS[1:]]
    training = cfg['options']['Training']
    training['epochs'] = max(int(training['epochs']), metadata['dgpo_next_epoch'] +
                            math.ceil((metadata['dgpo_epoch_step'] + 50) / cfg['dgpo']['steps_per_epoch']) + 2)
    name = f'Does conditioning retain reward? | {label} | 50 updates'
    if len(name) > 96:
        raise ValueError('W&B display name too long')
    cfg['logger']['wandb'].update(id=run_id, fresh_run=False, resume='never', run_name=name)
    cfg['logger']['local'].update(name=f'late-conditioning-{arm}', save_dir=str(output / 'logs'))
    cfg['experiment'].update(arm=arm, basis=basis, source_policy_step=metadata['global_step'],
        intervention='unchanged_native_control' if arm == 'A' else f'zero_output_{basis}_angle_velocity_residual',
        source_wandb_run='ytchou97-university-of-washington/nu2flow-RL/6704e92e',
        primary_endpoint='heldout_paired_C_minus_B_reward_gain_at_update50',
        endpoint_selection='predeclared_relative_step50',
        representation_note='No clean truth input. Inverse-normalized x_t defines noisy reconstructed angles, not clean physical estimates.',
        optimizer_note='Keep old groups, Adam moments and cosine clock; only new branch moments start empty.')
    cfg['experiment']['comparison_scope'] = (
        'C-versus-B changes only the angular basis at identical parameter count/initialization. '
        'B-versus-A tests the full added individual-angle residual package, not capacity alone.')
    cfg['nersc']['execution'] = {'command': 'python3 -u scripts/diagnose_dgpo_conditioning.py'}
    cfg['nersc']['reproducibility']['note'] = (
        'Full raw late-state continuation. Fixed installed classifier and serialized references. '
        'A saves update batches, B/C replay them with per-update paired RNG. '
        'Zero-output branch; original AdamW moments/LR schedule preserved. No audit/refit/EMA. '
        '50 updates is a new matched screen, not bitwise historical continuation.')
    validate_probe_runtime(cfg)
    return cfg


def prepare(args):
    import torch
    runtime = args.source_runtime.expanduser().resolve(strict=True)
    source = args.checkpoint.expanduser().resolve(strict=True)
    # Validate before allocating an output directory or loading a large checkpoint.
    original = read_yaml(runtime)
    if original['platform']['number_of_workers'] != 16:
        raise ValueError('Source runtime must use 16 workers')
    args.output_root.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix=datetime.now(timezone.utc).strftime('screen-%Y%m%dT%H%M%S-'),
                                dir=args.output_root.resolve()))
    pinned = root / 'source.ckpt'
    try:
        os.link(source, pinned)
    except OSError:
        shutil.copy2(source, pinned)
    state = torch.load(pinned, map_location='cpu', weights_only=False, mmap=True)
    meta = source_metadata(state)
    if any('visible_conditioning.reward_probe.' in k for k in state['state_dict']):
        raise ValueError('Choose the original stalled checkpoint, not a probe endpoint')
    del state
    old_probe = original['dgpo'].get('checkpoint_transfer', {})
    if (old_probe.get('source_step') != meta['global_step'] or
            old_probe.get('source_reward_round') != meta['dgpo_reward_round_id']):
        raise ValueError('Source runtime must describe the exact pinned late checkpoint/round')
    meta['source_checkpoint'] = str(source)
    source_runtime = root / 'source_runtime.yaml'
    shutil.copy2(runtime, source_runtime)
    plan = {'status': 'prepared', 'source': meta, 'steps': STEPS, 'world_size': 16,
            'comparison_run_id': uuid.uuid4().hex[:8], 'arms': {}}
    for arm in ARMS:
        output = root / arm
        output.mkdir()
        run_id = uuid.uuid4().hex[:8]
        cfg = build_arm_config(source_runtime, pinned=pinned, output=output, root=root,
                               metadata=meta, arm=arm, run_id=run_id)
        path = output / 'runtime.yaml'
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        plan['arms'][arm] = {'run_id': run_id, 'runtime': str(path), 'output': str(output)}
    (root / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps({'prepared': str(root), 'source_step': meta['global_step'],
        'reward_round': meta['dgpo_reward_round_id'], 'updates_per_arm': 50, 'arms': plan['arms'],
        'heldout_events': 32768, 'classifiers_fitted': 0, 'reference_coefficient': 1}, indent=2), flush=True)
    return root, plan


def load_scores(root, arm, step):
    arrays, ids = [], []
    for rank in range(16):
        with np.load(root / arm / 'measurements' / f'heldout_step{step:02d}_rank{rank:02d}.npz') as p:
            arrays.append(p['rewards'].astype(np.float64))
            ids.extend(p['event_ids'].tolist())
    scores = np.concatenate(arrays)
    if len(set(ids)) != len(ids) or scores.shape != (32768, 8) or not np.isfinite(scores).all():
        raise ValueError('Expected unique 32768-event K8 paired reward panel')
    return ids, scores


def retry_incomplete_arm(root, plan, arm):
    """Restart only an explicit unfinished B/C attempt, preserving A and evidence.

    This is NOT resuming the failed actor: replay again from the pinned source,
    with a fresh W&B ID and the same saved A batches/noise/held-out panel.
    """
    if arm not in ('B', 'C'):
        raise ValueError('Only B/C can restart against the completed A cache')
    control_report = root / 'A' / 'measurements' / 'report.json'
    control = json.loads(control_report.read_text())
    if control.get('status') != 'complete' or control.get('primary_endpoint') != 50:
        raise ValueError('Retry requires the completed 50-update A control')
    entry = plan['arms'][arm]
    output, runtime = Path(entry['output']), Path(entry['runtime'])
    if (output.is_symlink() or output.resolve() != (root / arm).resolve()
            or runtime.resolve() != (output / 'runtime.yaml').resolve()):
        raise ValueError('Retry must target this screen\'s exact arm directory')
    report_path = output / 'measurements' / 'report.json'
    report = json.loads(report_path.read_text())
    if report.get('status') == 'complete':
        raise ValueError(f'Arm {arm} is complete; keep the existing result')
    cfg = read_yaml(runtime)
    if (cfg['experiment']['arm'] != arm or
            cfg['dgpo']['checkpoint_transfer']['update_cache_mode'] != 'read'):
        raise ValueError('Retry configuration is not the requested cache-replay arm')
    run_id = uuid.uuid4().hex[:8]
    archive = root / f'{arm}.interrupted-{entry["run_id"]}-{run_id}'
    cfg['logger']['wandb'].update(id=run_id, fresh_run=False, resume='never',
        run_name=f'Does conditioning retain reward? | {ARMS[arm][0]} | 50 updates | Retry')
    cfg['logger']['wandb']['tags'] = list(dict.fromkeys([*cfg['logger']['wandb'].get('tags', []), 'Retry']))
    cfg['experiment']['retry_of_run'] = entry['run_id']
    cfg['experiment']['retry_mode'] = 'restart_from_pinned_source_with_original_A_cache'
    validate_probe_runtime(cfg)
    # Preserve the old logs, measurements and runtime rather than overwriting
    # a partial trajectory or appending step0 into its W&B history.
    output.rename(archive)
    output.mkdir()
    runtime.write_text(yaml.safe_dump(cfg, sort_keys=False))
    plan.setdefault('interrupted_attempts', []).append({
        'arm': arm, 'run_id': entry['run_id'], 'output': str(archive),
        'runtime': str(archive / 'runtime.yaml'), 'replaced_by_run_id': run_id})
    plan['arms'][arm] = {**entry, 'run_id': run_id}
    (root / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    print(f'Arm {arm}: preserved interrupted attempt at {archive}; '
          f'restarting from pinned source in new W&B run {run_id}. A is unchanged.', flush=True)


def simultaneous_intervals(vectors, replicates=2000, seed=20260929):
    """Event-cluster bootstrap with a max-standardized-deviation simultaneous band."""
    names = list(vectors)
    data = np.stack([vectors[k] for k in names], 1)
    if data.ndim != 2 or len(data) < 2 or not np.isfinite(data).all():
        raise ValueError('Invalid paired event contrast vectors')
    mean = data.mean(0)
    se = data.std(0, ddof=1) / np.sqrt(len(data))
    rng = np.random.default_rng(seed)
    draws = np.empty((replicates, len(names)))
    for i in range(replicates):
        draws[i] = data[rng.integers(len(data), size=len(data))].mean(0)
    scale = np.where(se > 0, se, 1.)
    critical = np.quantile(np.max(np.abs((draws - mean) / scale), axis=1), .95)
    return {k: {'mean': float(mean[i]), 'lo95_simultaneous': float(mean[i] - critical*se[i]),
                'hi95_simultaneous': float(mean[i] + critical*se[i])} for i, k in enumerate(names)}


def summarize(root, plan, *, wandb_enabled=True):
    scores, common = {}, None
    for arm in ARMS:
        for step in STEPS:
            ids, values = load_scores(root, arm, step)
            if common is None:
                common = ids
            if common != ids:
                raise ValueError('Paired identities or ordering differ between arms')
            scores[arm, step] = values
        if not np.allclose(scores[arm, 0], scores['A', 0], atol=1e-6, rtol=0):
            raise ValueError('Arms have different starting rewards')
    vectors = {}
    gains = {arm: (scores[arm, 50] - scores[arm, 0]).mean(1) for arm in ARMS}
    for arm in ARMS:
        vectors[f'{arm}/gain50'] = gains[arm]
        vectors[f'{arm}/change5_to50'] = (scores[arm, 50] - scores[arm, 5]).mean(1)
    vectors['B_minus_A/gain50'] = gains['B'] - gains['A']
    vectors['C_minus_B/gain50'] = gains['C'] - gains['B']
    vectors['C_minus_A/gain50'] = gains['C'] - gains['A']
    intervals = simultaneous_intervals(vectors)
    main = intervals['C_minus_B/gain50']
    supports = (main['mean'] > .01 and main['lo95_simultaneous'] > 0
                and intervals['C/gain50']['lo95_simultaneous'] > 0)
    report = {'status': 'complete', 'source': plan['source'], 'events': len(common), 'K': 8,
        'primary_endpoint': 'C_minus_B/gain50', 'contrasts': intervals,
        'decision': 'supports_relational_basis_in_this_screen' if supports else 'unresolved_or_no_material_advantage_in_50_updates',
        'scope': 'One matched trajectory; fixed reward. No fresh-classifier closure or training-seed uncertainty.',
        'uncertainty': 'Simultaneous 95% event-bootstrap bands across the nine predeclared contrasts.'}
    (root / 'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    if wandb_enabled:
        import wandb
        cfg = read_yaml(Path(plan['arms']['A']['runtime']))['logger']['wandb']
        with wandb.init(project=cfg['project'], entity=cfg.get('entity'), id=plan['comparison_run_id'],
                        resume='allow', name='Does conditioning retain reward? | Paired comparison | 50 updates',
                        group=cfg['group'], config={'protocol': 'late-reward-conditioning-50', 'source': plan['source']},
                        dir=str(root)) as run:
            run.define_metric('relative_update')
            run.define_metric('reward/*', step_metric='relative_update')
            for step in STEPS:
                run.log({'relative_update': step, **{f'reward/{a}_gain': float((scores[a, step]-scores[a, 0]).mean()) for a in ARMS}})
            run.summary.update({'decision': report['decision'], **{
                f'paired/{key}/{stat}': value for key, stats in intervals.items() for stat, value in stats.items()}})
            artifact = wandb.Artifact(f'conditioning-comparison-{plan["comparison_run_id"]}', type='diagnostic')
            artifact.add_file(str(root / 'comparison.json'))
            artifact.add_file(str(root / 'plan.json'))
            run.log_artifact(artifact)
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=SOURCE / 'source.ckpt')
    parser.add_argument('--source-runtime', type=Path, default=SOURCE / 'runtime.yaml')
    parser.add_argument('--output-root', type=Path, default=OUTPUT)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--prepared', type=Path, help='Execute an already prepared experiment directory')
    parser.add_argument('--retry-arm', choices=['B', 'C'],
                        help='Restart an unfinished arm from the pinned source; archive its old output and keep completed A')
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--no-wandb-report', action='store_true', help='Only suppress the final comparison upload')
    args = parser.parse_args()
    if args.report_only and args.prepared is None:
        parser.error('--report-only requires --prepared')
    if args.retry_arm and (args.prepared is None or args.report_only):
        parser.error('--retry-arm requires --prepared and cannot be combined with --report-only')
    if args.prepared:
        root = args.prepared.resolve(strict=True)
        plan = json.loads((root / 'plan.json').read_text())
    else:
        root, plan = prepare(args)
    if args.retry_arm:
        retry_incomplete_arm(root, plan, args.retry_arm)
    if args.prepare_only:
        return
    if not args.report_only:
        env = dict(os.environ)
        env['PYTHONPATH'] = os.pathsep.join([str(EVENET_DGPO_ROOT), env.get('PYTHONPATH', '')])
        for arm in ARMS:
            output = Path(plan['arms'][arm]['output'])
            report_path = output / 'measurements' / 'report.json'
            if report_path.exists():
                if json.loads(report_path.read_text()).get('status') == 'complete':
                    print(f'Arm {arm}: completed measurements already present; skipping', flush=True)
                    continue
                if arm == 'A':
                    raise RuntimeError('Arm A interrupted: prepare a fresh screen; its paired update cache is incomplete')
                raise RuntimeError(f'Arm {arm} interrupted: use --prepared {root} --retry-arm {arm} '
                                   'for an explicit restart of B/C; no silent restart')
            command = command_for_backend('dgpo-evenet', Path(plan['arms'][arm]['runtime'])) + [
                '--max-steps', str(plan['source']['global_step'] + 50), '--ray-dir', str(output / 'ray_results')]
            print(f'Starting arm {arm}: {ARMS[arm][0]} (50 additional updates)', flush=True)
            subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)
    summarize(root, plan, wandb_enabled=not args.no_wandb_report)


if __name__ == '__main__':
    main()
