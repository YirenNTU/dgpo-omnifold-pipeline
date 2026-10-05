"""Matched fresh tau heads for the step-1920 classifier-noise experiment.

All members use one K=1 negative panel and the same internal split. Member 0
is the predeclared single-head control. The ensemble averages training-time
bounded log ratios, before the unchanged native DGPO advantage and gate.
Fitting and temporary scoring never install a production head, advance reward
clocks, or access/recenter the velocity reference.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


SCHEMA_VERSION = 1
KIND = 'tau_classifier_ensemble_diagnostic'
SOURCE_STEP = 1920
INHERITED_DENOMINATOR_STEP = 1880
MEMBERS = 4
CONDITIONING_CONTRACT = 'saved_tau_denominator_label0_v1'
FIT_KEYS = {'lr', 'weight_decay', 'epochs', 'min_lr', 'batch_size', 'patience',
            'min_delta', 'min_steps'}
ARCHITECTURE_KEYS = ('condition_dim', 'candidate_dim', 'hidden', 'dropout',
    'head_kind', 'head_depth', 'condition_hidden', 'condition_width',
    'relative_dim', 'ratio_bound', 'ratio_objective', 'cross_attention',
    'attention_heads', 'packing_spec', 'relative_preprocessing')
CLOCK_KEYS = ('round_id', 'denominator_step', 'last_refit_epoch', 'last_evaluation_epoch')


def _serializable(value):
    """Keep artifacts weights_only-loadable and independent of live tensors."""
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _serializable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_serializable(v) for v in value]
    return copy.deepcopy(value)


def _equal(left, right):
    if isinstance(left, dict) or isinstance(right, dict):
        return (isinstance(left, dict) and isinstance(right, dict)
                and left.keys() == right.keys()
                and all(_equal(left[k], right[k]) for k in left))
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (isinstance(left, (list, tuple)) and isinstance(right, (list, tuple))
                and len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right)))
    if torch.is_tensor(left) or torch.is_tensor(right) or isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        try:
            return torch.equal(torch.as_tensor(left, device='cpu'), torch.as_tensor(right, device='cpu'))
        except (TypeError, ValueError):
            return False
    return left == right


def _reward_contract(reward):
    bundle = reward.bundle
    return _serializable({k: bundle[k] for k in ('source_run', 'source_checkpoint',
        'normalization_file', 'condition_mean', 'condition_scale')}) | {
        'feature_trunk': 'frozen_raw1110', 'policy_conditioning_contract': CONDITIONING_CONTRACT,
        'architecture': _serializable({k: reward.installed_head.get(k)
                                      for k in ARCHITECTURE_KEYS})}


def _validate_head(cfg):
    required = dict(head_kind='film', head_depth=3, condition_width=256,
        condition_hidden=256, relative_dim=6, ratio_bound=30,
        ratio_objective='bce', cross_attention=True, attention_heads=4)
    for key, value in required.items():
        if cfg.get(key, 'bce' if key == 'ratio_objective' else None) != value:
            raise ValueError(f'Ensemble must preserve attention tau head {key}={value}')
    if cfg.get('explicit_input') or 'classification' in cfg['packing_spec']['shapes']:
        raise ValueError('Ensemble requires the pinned raw1110 features and fixed label0 contract')


class MeanBoundedLogRatio(torch.nn.Module):
    """Arithmetic mean of the four bounded outputs, not latent/probability mean."""
    def __init__(self, members):
        super().__init__()
        if len(members) != MEMBERS:
            raise ValueError('The matched ensemble requires exactly four heads')
        self.members = torch.nn.ModuleList(members)

    def forward(self, condition, candidate):
        return torch.stack([head(condition, candidate) for head in self.members]).mean(0)


def ensemble_disagreement(scores, event_weight, *, tie_tolerance=1e-8):
    """Describe same-candidate preference noise; this is not update efficacy.

Input axes are (member,K,event). The advantage is exactly the native
leave_one_out_unscaled baseline, with no normalization, clipping or tilt.
Zero directions remain undefined (None), never reported as perfect agreement.
"""
    r, w = np.asarray(scores, dtype=np.float64), np.asarray(event_weight, dtype=np.float64)
    if (r.ndim != 3 or r.shape[0] != MEMBERS or r.shape[1] < 2
            or w.shape != (r.shape[2],) or not np.isfinite(r).all()
            or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0
            or not math.isfinite(tie_tolerance) or tie_tolerance < 0):
        raise ValueError('Require finite (4,K,B) scores and nonnegative positive-mass event weights')
    if (r > math.log(30)+1e-5).any():
        raise ValueError('Ensemble diagnostics require bounded log ratios')
    w = w/w.sum()
    k = r.shape[1]
    advantage = k/(k-1)*(r-r.mean(1, keepdims=True))
    sign = lambda x: np.where(np.abs(x) <= tie_tolerance, 0, np.sign(x))
    average = lambda x: float(w @ np.asarray(x).mean(axis=0))
    a, b = np.triu_indices(k, 1)
    preference = sign(r[:, a]-r[:, b])
    pairs = {}
    for i in range(MEMBERS):
        for j in range(i+1, MEMBERS):
            pi, pj = preference[i], preference[j]
            comparable = (pi != 0) & (pj != 0)
            mass = average(comparable)
            disagreement = average((pi != pj) & comparable)
            ai, aj = advantage[i], advantage[j]
            dot = average(ai*aj)
            norm = math.sqrt(average(ai*ai)*average(aj*aj))
            pairs[f'{i}-{j}'] = dict(
                ranking_disagreement=disagreement/mass if mass else None,
                ranking_comparable_fraction=mass,
                ranking_tie_mismatch_fraction=average((pi == 0) != (pj == 0)),
                advantage_sign_disagreement=average(sign(ai) != sign(aj)),
                advantage_opposite_sign_fraction=average(sign(ai)*sign(aj) < 0),
                advantage_cosine=dot/norm if norm else None,
                advantage_rms_difference=math.sqrt(average((ai-aj)**2)))
    mean = advantage.mean(0)
    single_rms = math.sqrt(average(advantage[0]**2))
    ensemble_rms = math.sqrt(average(mean**2))
    centered_variance = average(advantage.var(0))
    return dict(members=MEMBERS, candidates=k, events=r.shape[2], single_control_member=0,
        aggregation='mean_training_time_bounded_log_ratio',
        advantage_estimator='leave_one_out_unscaled', pairwise=pairs,
        member_advantage_rms=[math.sqrt(average(x*x)) for x in advantage],
        single_advantage_rms=single_rms, ensemble_advantage_rms=ensemble_rms,
        ensemble_to_single_advantage_rms=ensemble_rms/single_rms if single_rms else None,
        centered_member_variance=centered_variance,
        limitation='Agreement/variance describe classifier uncertainty, not independent reward or spin closure; shared bias remains possible')


class TauClassifierEnsemble:
    def __init__(self, payload, reward, device):
        from scripts.train_conditional_spin_ratio import build_classifier
        from scripts.tau_fresh_negatives import isolated_rng
        self.payload = payload
        self.device = torch.device(device)
        if (payload.get('kind') != KIND or payload.get('schema_version') != SCHEMA_VERSION
                or not payload.get('complete')
                or payload.get('source_policy_step') != SOURCE_STEP
                or payload.get('single_control_member') != 0
                or payload.get('aggregation') != 'mean_training_time_bounded_log_ratio'
                or len(payload.get('members', [])) != MEMBERS
                or 'judge' not in payload):
            raise ValueError('Invalid step1920 ensemble artifact')
        self._assert_reward(reward)
        heads = []
        all_saved = [*payload['members'], payload['judge']]
        seeds = [member['seed'] for member in all_saved]
        if len(set(seeds)) != MEMBERS+1:
            raise ValueError('Judge and four members require independent initializations')
        for member in all_saved:
            saved = member['head']
            _validate_head(saved)
            if (member['seed'] != saved.get('seed')
                    or saved.get('source_policy_step') != SOURCE_STEP
                    or saved.get('training_candidates') != 1):
                raise ValueError('Ensemble/judge member lost its independent seed or pinned K1 denominator')
            status = member['fit']
            if (not status.get('minimum_fit_steps_met')
                    or status['total_steps'] < max(1000, saved['min_steps'])):
                raise ValueError('Ensemble/judge artifact is undertrained')
            if not _equal({k: saved.get(k) for k in ARCHITECTURE_KEYS}, payload['reward_contract']['architecture']):
                raise ValueError('Ensemble member changed architecture/preprocessing')
            with isolated_rng(self.device, int(saved['seed'])):
                head = build_classifier(saved).to(self.device)
                head.load_state_dict(saved['state_dict'], strict=True)
            heads.append(head.eval().requires_grad_(False))
        self.heads = tuple(heads[:MEMBERS])
        self.judge_head = heads[MEMBERS]
        self.mean_head = MeanBoundedLogRatio(self.heads).to(self.device).eval().requires_grad_(False)

    def _assert_reward(self, reward):
        if not _equal(_reward_contract(reward), self.payload['reward_contract']):
            raise ValueError('Ensemble source bundle, normalization or head architecture changed')
        diagnostic = getattr(reward, '_tau_diagnostic_ensemble', None)
        if diagnostic is not None:
            # A persistent head-only arm has a real fresh1920 denominator. Its
            # teacher and independent judge still come from the same saved fit.
            if not _equal(diagnostic['artifact'], self.payload):
                raise ValueError('Installed diagnostic teacher comes from a different ensemble artifact')
            for key in CLOCK_KEYS:
                if getattr(reward, key, None) != diagnostic['installed_clocks'][key]:
                    raise ValueError('Installed diagnostic reward clock changed: '+key)
            return
        for key in CLOCK_KEYS:
            if getattr(reward, key, None) != self.payload['inherited_clocks'][key]:
                raise ValueError('Ensemble must load against the same inherited reward clocks: '+key)

    @classmethod
    def load(cls, path, reward, device):
        path = Path(path)
        return cls(torch.load(path/'ensemble.pt' if path.is_dir() else path,
                   map_location='cpu', weights_only=True), reward, device)

    def head_for(self, selection):
        if selection == 'ensemble':
            return self.mean_head
        if selection == 'judge':
            return self.judge_head
        if selection in ('single', 'member0'):
            return self.heads[0]
        if selection in ('member1', 'member2', 'member3'):
            return self.heads[int(selection[-1])]
        raise ValueError('Select predeclared single/member0, ensemble, member1..3, or evaluation judge')

    def install_into_reward(self, reward, selection='single', *, policy_step=SOURCE_STEP, epoch=191):
        """Commit a declared teacher intervention without recentering reference.

This is used only by the fresh-single/ensemble trajectory arms, after fitting
and fixed-candidate diagnosis have finished. The inherited arm needs no call.
The fixed judge is never a training teacher.
"""
        self._assert_reward(reward)
        if selection not in ('single', 'ensemble') or policy_step != SOURCE_STEP or type(epoch) is not int:
            raise ValueError('Persistent teacher must be fresh single or ensemble at policy1920')
        if getattr(reward, '_tau_diagnostic_ensemble', None) is not None:
            raise ValueError('The diagnostic teacher may only be installed once per source restore')
        clocks = {k: getattr(reward, k, None) for k in CLOCK_KEYS}
        clocks.update(round_id=int(reward.round_id)+1, denominator_step=policy_step, last_refit_epoch=epoch)
        installed_head = copy.deepcopy(self.payload['members'][0]['head'])
        installed_head.update(diagnostic_ensemble_selection=selection,
                              diagnostic_generator_checkpoint=self.payload['generator_checkpoint'])
        reward.head = self.head_for(selection)
        reward.installed_head = installed_head
        for key, value in clocks.items():
            setattr(reward, key, value)
        reward._tau_diagnostic_ensemble = dict(artifact=self.payload, selection=selection,
                                              installed_clocks=clocks)
        reward._tau_diagnostic_ensemble_owner = self

    @contextmanager
    def temporary_reward(self, reward, selection='single'):
        """A reversible head swap; reward/reference clocks stay inherited1880."""
        self._assert_reward(reward)
        head = self.head_for(selection)
        original = {key: getattr(reward, key, None) for key in ('head', 'installed_head', *CLOCK_KEYS)}
        reward.head = head
        # Keep installed_head's production provenance unchanged. The fresh1920
        # denominator is explicit in this artifact, not a fake production refit.
        try:
            yield head
        finally:
            changed = [key for key in CLOCK_KEYS if getattr(reward, key, None) != original[key]]
            for key, value in original.items():
                setattr(reward, key, value)
            if changed:
                raise RuntimeError('Diagnostic changed reward clocks: '+', '.join(changed))

    @torch.no_grad()
    def score_candidates(self, reward, batch, candidates, mask=None, *, include_judge=False):
        from RL.DGPO_neutrino.conditional_tau_reward import feature_precision
        from RL.DGPO_neutrino.rewards import apply_event_valid_to_rewards
        self._assert_reward(reward)
        if candidates.ndim != 4 or candidates.shape[-2:] != (2, 2) or candidates.shape[0] < 2:
            raise ValueError('Ensemble scoring expects shared (K,B,2,2) candidates, K>=2')
        values = []
        with feature_precision():
            for delta in candidates:
                c, f = reward.features(batch, delta)
                condition, feature = [torch.as_tensor(x, device=self.device) for x in (c, f)]
                heads = (*self.heads, self.judge_head) if include_judge else self.heads
                values.append(torch.stack([head(condition, feature) for head in heads]))
        result = torch.stack(values, dim=1)
        if not torch.isfinite(result).all() or bool((result > math.log(30)+1e-5).any()):
            raise ValueError('Nonfinite/unbounded ensemble output')
        if mask is not None:
            result = result*mask.to(result)
        return torch.stack([apply_event_valid_to_rewards(member, batch) for member in result])


def diagnostic_stack_payload(reward):
    """Serialize all frozen members, and prohibit saving a temporary judge."""
    diagnostic = getattr(reward, '_tau_diagnostic_ensemble', None)
    if diagnostic is None:
        return None
    owner = reward._tau_diagnostic_ensemble_owner
    owner._assert_reward(reward)
    if reward.head is not owner.head_for(diagnostic['selection']):
        raise ValueError('Cannot checkpoint a temporary diagnostic/judge head as a trajectory teacher')
    head = _serializable(reward.installed_head)
    # The normal head field remains a valid member0 architecture/checkpoint.
    # The explicit diagnostic field carries all four members and aggregation.
    head['state_dict'] = _serializable(owner.heads[0].state_dict())
    return dict(head=head, diagnostic_ensemble=_serializable(diagnostic))


def restore_diagnostic_reward(reward, saved):
    """Validate/build completely before replacing a resumed diagnostic teacher."""
    diagnostic = saved['diagnostic_ensemble']
    payload, selection = diagnostic['artifact'], diagnostic['selection']
    if selection not in ('single', 'ensemble'):
        raise ValueError('A fixed judge/member diagnostic cannot be resumed as a training teacher')
    clocks = {key: int(saved[key]) for key in CLOCK_KEYS}
    expected = dict(payload['inherited_clocks'],
        round_id=int(payload['inherited_clocks']['round_id'])+1,
        denominator_step=SOURCE_STEP,
        last_refit_epoch=diagnostic['installed_clocks']['last_refit_epoch'])
    if not _equal(clocks, diagnostic['installed_clocks']) or not _equal(clocks, expected):
        raise ValueError('Resumed ensemble reward clocks are inconsistent with its head-only install')
    member0 = payload['members'][0]['head']
    if (not _equal(saved['head']['state_dict'], member0['state_dict'])
            or saved['head'].get('diagnostic_ensemble_selection') != selection):
        raise ValueError('Resumed member0 checkpoint disagrees with the saved diagnostic ensemble')
    view = SimpleNamespace(bundle=reward.bundle, installed_head=saved['head'],
                           **payload['inherited_clocks'])
    ensemble = TauClassifierEnsemble(payload, view, reward.device)
    reward.head = ensemble.head_for(selection)
    reward.installed_head = copy.deepcopy(saved['head'])
    for key, value in clocks.items():
        setattr(reward, key, value)
    reward._tau_diagnostic_ensemble = copy.deepcopy(diagnostic)
    reward._tau_diagnostic_ensemble_owner = ensemble


def fit_tau_ensemble(cycle, generated, folder, *, step=SOURCE_STEP,
                     seeds=(202610041, 202610042, 202610043, 202610044),
                     judge_seed=202610045,
                     source_checkpoint=None, fit_overrides=None, allow_cpu_fixture=False):
    """Fit-only transaction on one current-policy negative panel, all 16 ranks."""
    from RL.DGPO_neutrino.conditional_tau_cycle import fit_head, barrier
    from RL.DGPO_neutrino.conditional_tau_reward import feature_precision
    from scripts.tau_fresh_negatives import isolated_rng
    if cycle.world != 16 and not allow_cpu_fixture:
        raise ValueError('Real-case ensemble fitting requires the existing 16-GPU group')
    if not allow_cpu_fixture and (not source_checkpoint or not isinstance(source_checkpoint, str)):
        raise ValueError('Record the pinned policy1920 checkpoint path in the ensemble artifact')
    if step != SOURCE_STEP or cycle.reward.denominator_step != INHERITED_DENOMINATOR_STEP:
        raise ValueError('Ensemble requires policy1920 and inherited denominator1880')
    if len(seeds) != MEMBERS or len(set(seeds)) != MEMBERS or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError('Predeclare four unique nonnegative integer member seeds')
    if type(judge_seed) is not int or judge_seed < 0 or judge_seed in seeds:
        raise ValueError('Judge seed must be independent of the four ensemble members')
    _validate_head(cycle.reward.installed_head)
    overrides = fit_overrides or {}
    if set(overrides)-FIT_KEYS:
        raise ValueError('Ensemble overrides may change fit budget, not architecture or split')
    panel = cycle.train
    if not allow_cpu_fixture and (len(panel['source_ids']) != 416701 or len(cycle.validation['source_ids']) != 119002):
        raise ValueError('Use the complete filtered train and independent validation panels')
    if np.intersect1d(panel['source_ids'], cycle.validation['source_ids']).size:
        raise ValueError('Ensemble fit and external validation identities overlap')
    arrays = {k: panel[k] for k in ('source_ids', 'condition', 'candidate_truth', 'event_weight', 'split')}
    arrays['candidate_generated'] = generated['features']
    if (np.shape(arrays['candidate_generated']) != np.shape(arrays['candidate_truth'])
            or set(np.unique(arrays['split'])) != {0, 1}
            or ('deltas' in generated and np.shape(generated['deltas'])[1:] != (1, 2, 2))):
        raise ValueError('Ensemble requires identical K1 negatives and the original two-way internal split')
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    inherited = {k: getattr(cycle.reward, k, None) for k in CLOCK_KEYS}
    old_head, old_installed = cycle.reward.head, cycle.reward.installed_head
    cfg = copy.deepcopy(cycle.reward.installed_head)
    cfg.pop('state_dict', None)
    cfg.update({k: v for k, v in cycle.cfg['fit'].items() if k in FIT_KEYS})
    cfg.update(overrides)
    cfg.update(ratio_objective='bce', ratio_bound=30, fitting_current_policy=True,
               source_policy_step=step, training_candidates=1)
    if cfg['min_steps'] < 1000:
        raise ValueError('Every ensemble member and fixed judge requires at least 1000 fit updates')
    members = []
    judge = None
    for member, seed in enumerate((*seeds, judge_seed)):
        member_cfg = dict(cfg, seed=seed)
        label = f'member{member}' if member < MEMBERS else 'judge'
        output = folder/f'member-{member:02d}' if member < MEMBERS else folder/'judge'
        output.mkdir(parents=True, exist_ok=True)
        def emit(row, label=label):
            cycle.emit({**{f'tau/ensemble/{label}/fit/{k}': v for k, v in row.items()},
                        'tau/ensemble/policy_step': step, 'tau/ensemble/actor_updates': 0}, step)
        with isolated_rng(cycle.device, seed), feature_precision():
            model, best, status = fit_head(member_cfg, arrays, output/'best.pt',
                                          cycle.device, cycle.rank, cycle.world, emit)
        del model
        if not status['minimum_fit_steps_met'] or status['total_steps'] < max(1000, cfg['min_steps']):
            raise ValueError('Ensemble member did not meet its predeclared fit budget')
        saved_member = dict(index=member, seed=seed, head=_serializable(best), fit=status)
        if member < MEMBERS:
            members.append(saved_member)
        else:
            judge = saved_member
        if (cycle.reward.head is not old_head or cycle.reward.installed_head is not old_installed
                or any(getattr(cycle.reward, k, None) != v for k, v in inherited.items())):
            raise RuntimeError('Fit-only ensemble changed the production reward')
    payload = dict(kind=KIND, schema_version=SCHEMA_VERSION, complete=True,
        source_policy_step=step, generator_checkpoint=source_checkpoint,
        fresh_denominator_step=step, inherited_denominator_step=INHERITED_DENOMINATOR_STEP,
        inherited_clocks=inherited, reward_contract=_reward_contract(cycle.reward),
        single_control_member=0, aggregation='mean_training_time_bounded_log_ratio',
        selection='Each member: absolute minimum internal-validation BCE; single control: member0, never best-of4',
        training_candidates=1, split='Original panel split shared by all members',
        training_population=len(panel['source_ids']), members=members, judge=judge,
        judge_role='Independent initialization; fixed common evaluator for every arm; excluded from ensemble/reward updates',
        judge_limitation='Judge shares the classifier fitting population/internal split; fresh final audit and held-out spin/Cij are separate endpoints',
        production_head_installed=False, reference_recentered=False, actor_updates=0)
    if cycle.rank == 0:
        pending = folder/'ensemble.pending.pt'
        torch.save(payload, pending); pending.replace(folder/'ensemble.pt')
        torch.save(judge, folder/'judge.pt')
        summary = {k: v for k, v in payload.items() if k not in ('members', 'judge', 'reward_contract')}
        summary['members'] = [{k: v for k, v in row.items() if k != 'head'} for row in members]
        summary['judge'] = {k: v for k, v in judge.items() if k != 'head'}
        (folder/'report.json').write_text(json.dumps(summary, indent=2)+'\n')
    barrier(cycle.world)
    return TauClassifierEnsemble.load(folder, cycle.reward, cycle.device)


def score_saved_ensemble_panel(cycle, ensemble, generated, candidate_folder, output, *, step=SOURCE_STEP):
    """Score one saved K8 validation panel with shared raw1110 features per draw."""
    from RL.DGPO_neutrino.conditional_tau_cycle import panel_batch, barrier
    from RL.DGPO_neutrino.conditional_tau_reward import feature_precision
    from scripts.train_conditional_spin_ratio import score_pair, pair_metrics
    output, candidate_folder = Path(output), Path(candidate_folder)
    output.mkdir(parents=True, exist_ok=True)
    ii = np.arange(cycle.rank, len(cycle.validation['source_ids']), cycle.world)
    with np.load(candidate_folder/f'rank-{cycle.rank:02d}.npz', allow_pickle=False) as saved:
        if not np.array_equal(saved['positions'], ii) or saved['deltas'].shape != (len(ii), 8, 2, 2):
            raise ValueError('Ensemble comparison requires identical validation identities and K8 candidates')
        deltas = saved['deltas']
    positive, negative = [], []
    with feature_precision():
        for head in (*ensemble.heads, ensemble.judge_head):
            p, q = score_pair(head, cycle.validation['condition'][ii],
                cycle.validation['candidate_truth'][ii], generated['features'][ii],
                cycle.device, cycle.cfg['fit']['batch_size'])
            positive.append(p); negative.append(q)
    scores = []
    for start in range(0, len(ii), cycle.cfg['generation_batch_size']):
        take = ii[start:start+cycle.cfg['generation_batch_size']]
        batch = panel_batch(cycle.validation, take, cycle.reward.spec, cycle.device)
        delta = torch.as_tensor(deltas[start:start+len(take)], device=cycle.device).permute(1, 0, 2, 3)
        scores.append(ensemble.score_candidates(cycle.reward, batch, delta, include_judge=True).permute(0, 2, 1).cpu().numpy())
    np.savez(output/f'rank-{cycle.rank:02d}.npz', positions=ii,
        source_ids=cycle.validation['source_ids'][ii], rewards=np.concatenate(scores, axis=1),
        truth_logits=np.stack(positive), generated_logits=np.stack(negative))
    barrier(cycle.world)
    report = None
    if cycle.rank == 0:
        n = len(cycle.validation['source_ids'])
        scores = np.empty((MEMBERS+1, n, 8)); p = np.empty((MEMBERS+1, n)); q = np.empty_like(p)
        seen = np.zeros(n, int)
        for rank in range(cycle.world):
            with np.load(output/f'rank-{rank:02d}.npz', allow_pickle=False) as saved:
                pos = saved['positions']; np.add.at(seen, pos, 1)
                if not np.array_equal(saved['source_ids'], cycle.validation['source_ids'][pos]):
                    raise ValueError('Ensemble scoring changed external source identities')
                scores[:, pos], p[:, pos], q[:, pos] = saved['rewards'], saved['truth_logits'], saved['generated_logits']
        if not np.all(seen == 1):
            raise ValueError('Ensemble score shards have missing/duplicate identities')
        weight = cycle.validation['event_weight']
        report = ensemble_disagreement(scores[:MEMBERS].transpose(0, 2, 1), weight)
        report.update(policy_step=step, actor_updates=0, reference_recentered=False,
            fresh_denominator_step=SOURCE_STEP, inherited_denominator_step=INHERITED_DENOMINATOR_STEP,
            external_members=[pair_metrics(a, b, weight) for a, b in zip(p[:MEMBERS], q[:MEMBERS])],
            external_ensemble=pair_metrics(p[:MEMBERS].mean(0), q[:MEMBERS].mean(0), weight),
            external_judge=pair_metrics(p[MEMBERS], q[MEMBERS], weight),
            judge_limitation=ensemble.payload['judge_limitation'])
        (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        np.savez_compressed(output/'measurements.npz', source_ids=cycle.validation['source_ids'],
            member_log_ratio=scores[:MEMBERS], single_log_ratio=scores[0], ensemble_log_ratio=scores[:MEMBERS].mean(0),
            judge_log_ratio=scores[MEMBERS], event_weight=weight,
            truth_logits=p[:MEMBERS], generated_logits=q[:MEMBERS],
            judge_truth_logits=p[MEMBERS], judge_generated_logits=q[MEMBERS])
        cycle.emit({'tau/ensemble/centered_member_variance': report['centered_member_variance'],
                    'tau/ensemble/ensemble_to_single_advantage_rms': report['ensemble_to_single_advantage_rms'],
                    'tau/ensemble/actor_updates': 0}, step)
    barrier(cycle.world)
    return report
