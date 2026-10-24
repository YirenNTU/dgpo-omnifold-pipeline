"""Opt-in W&B view for the corrected epoch287 General+Spin+Tau policy run.

This module only selects/aggregates monitoring data. It never changes samples,
losses, classifier selection, validation cadence, or saved physics reports.
"""
import numpy as np

PROFILE = 'tau_spin'
CIJ = 'tau/tihsu_reconstruction/cij'
AXES = ('n', 'r', 'k')
COMPONENTS = tuple(a+b for a in AXES for b in AXES)
OBSERVABLES = ('topology/cos_opening', 'topology/delta_phi_to_pi', *(
    f'target/tau_{leg}_delta_{angle}' for leg in ('a', 'b') for angle in ('theta', 'phi')))
FIT_KEYS = frozenset(('train_bce', 'val_bce', 'val_auc', 'best_val_bce', 'lr',
                     'grad_norm', 'early_stopped', 'epoch', 'optimizer_steps',
                     'log_index', 'policy_step', 'reward_round'))
STATUS_KEYS = frozenset(('total_steps', 'epochs', 'best_epoch', 'best_steps',
                        'best_val_bce', 'minimum_fit_steps_met', 'early_stopped'))
TRAIN_KEYS = frozenset((
    'train/loss/total', 'train/loss/dgpo', 'train/grad/global_norm_pre_clip',
    'train/grad/clip_active', 'train/parameter_update_rms', 'train/optimizer_step_ran',
    'train/memory/peak_allocated_gib', 'train/memory/peak_reserved_gib',
    'train/time/policy_step_seconds', 'train/reward_round', 'train/reward_age_updates',
    'reward/raw/mean', 'reward/raw/std', 'reward/advantage/std',
    'reward/advantage/positive_fraction', 'reward/monitor/advantage_pos_neg_gap',
    'reference_trust/loss', 'reference_trust/coefficient',
    'reference_trust/velocity_mse', 'reference_trust/velocity_mse_ratio',
    'train/visible_conditioning/met_scale_rms', 'train/visible_conditioning/met_shift_rms',
    'train/visible_conditioning/met_conditioning_grad_norm_post_clip',
))
PHYSICS_KEYS = frozenset((
    *(f'{CIJ}/{k}' for k in ('baseline', 'current', 'delta', 'change_ci_low',
                            'change_ci_high', 'channel_components')),
    *(f'{CIJ}/event_weighted_abs_error/{k}' for k in COMPONENTS),
    'tau/marginal/baseline_available', 'tau/marginal/plots/target',
    'tau/marginal/plots/topology',
    *(f'tau/marginal/{metric}/{arm}/{name}'
      for metric in ('tv', 'jsd') for arm in ('current', 'change') for name in OBSERVABLES),
    *(f'tau/marginal/tail_mass/{arm}/{name}'
      for arm in ('truth', 'initial', 'current') for name in OBSERVABLES),
))
REWARD_KEYS = frozenset(f'tau/reward/{k}' for k in (
    'mean', 'within_condition_std', 'within_condition_ess_fraction', 'near_cap_fraction'))


def keep(key):
    if key in TRAIN_KEYS | PHYSICS_KEYS | REWARD_KEYS or key in ('global_step', 'epoch', 'tau/reward_round'):
        return True
    if key.startswith('train/lr/scheduled/') or (key.startswith('train/') and 'nonfinite' in key):
        return True
    if key.startswith(CIJ+'/channel_') and key.endswith('/matched_error'):
        return True  # Searchable per-channel curves; hidden by default.
    if key.startswith('tau/startup/'):
        return True  # Replay/conditioning integrity flags, hidden by default.
    for phase in ('audit_fit', 'refit_fit'):
        if key.startswith(f'tau/{phase}/'):
            return key.removeprefix(f'tau/{phase}/') in FIT_KEYS
    if key.startswith('tau/fresh_audit/'):
        return key.removeprefix('tau/fresh_audit/') in STATUS_KEYS | {'auc', 'auc_gap', 'bce', 'general_only'}
    if key.startswith('tau/refit/'):
        return key.removeprefix('tau/refit/') in STATUS_KEYS | {
            'policy_step', 'reward_round', 'reference_recentered', 'actor_optimizer_reset', 'seconds'}
    if key.startswith('tau/validation/'):
        return key.removeprefix('tau/validation/') in {'events', 'candidates', 'seconds'}
    return False


def filter_payload(payload):
    return {str(k): v for k, v in payload.items() if keep(str(k))}


def define_axes(wandb):
    # All physics/training uses completed policy updates, not W&B row count.
    wandb.define_metric('*', step_metric='global_step', step_sync=False, hidden=False)
    for key in ('global_step', 'epoch', 'tau/startup/*', 'tau/validation/events',
                'tau/validation/candidates', 'tau/marginal/tail_mass/*',
                'tau/fresh_audit/general_only'):
        wandb.define_metric(key, hidden=True)
    # Use supported trailing globs (no embedded wildcard) for channel metadata.
    for a in range(1, 5):
        for b in range(1, 5):
            wandb.define_metric(f'{CIJ}/channel_{a}{b}/*', hidden=True)
    wandb.define_metric(CIJ+'/channel_components', hidden=False)
    for phase in ('audit_fit', 'refit_fit'):
        prefix = f'tau/{phase}'
        wandb.define_metric(prefix+'/*', step_metric=prefix+'/log_index', step_sync=False, hidden=False)
        for key in ('log_index', 'policy_step', 'reward_round', 'epoch', 'early_stopped'):
            wandb.define_metric(prefix+'/'+key, hidden=True)


def summary_metadata():
    return {
        'dgpo_logging_profile': PROFILE,
        'dgpo_clock_schema': 'completed-policy-steps-v2',
        'monitoring/cij': 'Pre-unfolding channel Cij; root-angle-exporter-target-p4-v2. Not unfolded uncertainty or concurrence.',
        'monitoring/cij_error': 'Event-mass-weighted mean of channel Frobenius errors; delta/current/CI compare native K8 samples to fixed step0.',
        'monitoring/cij_components': 'event_weighted_abs_error = weighted mean absolute channel bias, not a combined Cij or significance. All channel matrices in channel_components table.',
        'monitoring/topology': 'Fixed step0 versus current candidate0 and true-offset target, event weights only. TV/JSD include tails; separate from calibrated Spin observables.',
        'monitoring/reward': 'raw = bounded classifier log density ratio, not advantage. Advantage is unchanged leave-one-out. Scores across refits use different heads; compare within reward_round.',
        'monitoring/ess': 'ESS from per-event softmax scores is diagnostic only; policy training does not use those weights.',
        'monitoring/near_cap': 'Validation fraction with exp(log ratio) >= 0.9*cap; cap is an upper odds bound only, not symmetric reward clipping.',
        'monitoring/reference': 'Velocity MSE penalty, coefficient1; not exact endpoint KL. Reference recenters at each classifier refit.',
        'monitoring/audit': 'Fresh General audit, never installed. Check best_steps >=1000 and minimum_fit_steps_met before interpreting AUC. Fit curves use phase log_index; policy_step identifies independent fits.',
        'monitoring/details': 'Full local JSON/NPZ diagnostics are preserved; legacy Cij, reweight probes and inactive ablations are omitted from W&B.',
    }


def compact_cij_metrics(report, panel, wandb):
    """Nine bias curves plus one table retain channel signs without 288 panels."""
    axes = report['axes']
    weights = np.asarray(panel['event_weight'], dtype=float)
    category = np.asarray(panel['category'])
    total = weights.sum()
    errors = np.zeros((3, 3), dtype=float)
    rows, out = [], {}
    for channel, result in report['channels'].items():
        if 'error' not in result:
            continue
        mass = weights[category == int(channel)].sum()/total
        delta = np.asarray(result['generated']['Cij']) - result['matched_target']['Cij']
        errors += mass*np.abs(delta)
        out[f'{CIJ}/channel_{channel}/matched_error'] = result['error']
        for i, left in enumerate(axes):
            for j, right in enumerate(axes):
                rows.append([str(channel), left+right, int(result['events']), float(mass),
                             float(result['matched_target']['Cij'][i][j]),
                             float(result['generated']['Cij'][i][j]), float(delta[i, j])])
    for i, left in enumerate(axes):
        for j, right in enumerate(axes):
            out[f'{CIJ}/event_weighted_abs_error/{left}{right}'] = float(errors[i, j])
    out[CIJ+'/channel_components'] = wandb.Table(
        columns=['channel', 'component', 'events', 'event_mass', 'target', 'prediction', 'prediction_minus_target'],
        data=rows)
    return out
