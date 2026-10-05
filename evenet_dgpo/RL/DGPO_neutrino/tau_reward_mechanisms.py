"""Opt-in, original-state tau reward mechanism experiments inside 16 DGPO workers.

Diagnosis is a transaction: no optimizer/scheduler clock or actor/reference
change survives it. Signed component perturbations match PARAMETER RMS, not
function-space distance. They test local transfer, not a proposed new loss.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager, nullcontext
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist

from RL.DGPO_neutrino.local_rng import seeded_torch_rng, model_cuda_devices
from RL.DGPO_neutrino.tau_mechanism_gradients import NativeGradientAccumulator, summarize_replicas
from RL.DGPO_neutrino.tau_reward_transfer import paired_statistics, reward_statistics, _frozen_state


def _cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return copy.deepcopy(value)


@contextmanager
def native_transaction(actor, optimizer, *, seed):
    """Full rollback, including buffers, optimizer, mixed modes, grads and RNG."""
    params = tuple(actor.parameters())
    saved_params = [value.detach().cpu().clone() for value in params]
    saved_grads = [None if p.grad is None else p.grad.detach().cpu().clone() for p in params]
    buffers = [(value, value.detach().cpu().clone()) for value in actor.buffers()]
    modes = [(module, module.training) for module in actor.modules()]
    state = _cpu_copy(optimizer.state_dict())
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with seeded_torch_rng(seed, model_cuda_devices(actor)):
            random.seed(seed); np.random.seed(seed % 2**32)
            yield
    finally:
        with torch.no_grad():
            for p, original in zip(params, saved_params, strict=True):
                p.copy_(original.to(p))
            for value, original in buffers:
                value.copy_(original.to(value))
        # The production wrapper preserves the *current* configured WD on
        # load. Reset it first so a rollback also restores a temporary WD edit.
        saved_groups = state.get('optimizer', state)['param_groups']
        for live, saved in zip(optimizer.param_groups, saved_groups, strict=True):
            live['weight_decay'] = saved['weight_decay']
        optimizer.load_state_dict(state)
        for p, grad in zip(params, saved_grads, strict=True):
            p.grad = None if grad is None else grad.to(p)
        for module, training in modes:
            module.training = training
        random.setstate(python_state); np.random.set_state(numpy_state)


def parameter_vector(parameters):
    return torch.cat([p.detach().reshape(-1).cpu().float() for p in parameters])


def add_displacement(parameters, delta):
    if delta.ndim != 1 or delta.numel() != sum(p.numel() for p in parameters):
        raise ValueError('Displacement does not match the trainable parameter layout')
    if not bool(torch.isfinite(delta).all()):
        raise ValueError('Displacement must contain finite values')
    offset = 0
    with torch.no_grad():
        for p in parameters:
            p.add_(delta[offset:offset+p.numel()].reshape(p.shape).to(p))
            offset += p.numel()
    if offset != delta.numel():
        raise ValueError('Displacement does not match the trainable parameter layout')


def adamw_proposal(actor, optimizer, parameters, gradient, present, *, clip_norm, seed):
    """Actual inherited AdamW one-step displacement, always rolled back."""
    with native_transaction(actor, optimizer, seed=seed):
        before = parameter_vector(parameters)
        optimizer.zero_grad(set_to_none=True)
        offset = 0
        for p, used in zip(parameters, present, strict=True):
            value = gradient[offset:offset+p.numel()].reshape(p.shape)
            p.grad = value.to(p).clone() if used else None
            offset += p.numel()
        if offset != gradient.numel():
            raise ValueError('Gradient parameter layout changed')
        torch.nn.utils.clip_grad_norm_(parameters, clip_norm, error_if_nonfinite=True)
        optimizer.step()
        return parameter_vector(parameters) - before


def normalized_descent(gradient, rms):
    if not torch.isfinite(gradient).all() or not np.isfinite(rms) or rms <= 0:
        raise ValueError('Finite gradient and positive reference displacement RMS required')
    norm = float(gradient.double().square().mean().sqrt())
    return None if norm <= 1e-16 else -gradient * (rms / norm)


@contextmanager
def matched_sampling(seed, *, device=None):
    """Couple native rollout/dropout/timestep draws without rolling back AdamW."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    device = torch.device(device) if device is not None else torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        with seeded_torch_rng(seed, devices):
            random.seed(seed); np.random.seed(seed % 2**32)
            yield
    finally:
        random.setstate(python_state); np.random.set_state(numpy_state)


def parameter_blocks(actor):
    blocks = dict(conditioning=[], generation=[], backbone=[])
    offset = 0
    for name, p in actor.named_parameters():
        if not p.requires_grad:
            continue
        key = ('conditioning' if 'visible_conditioning.' in name or 'angular_conditioning.' in name else
               'generation' if 'TruthGeneration.' in name else 'backbone')
        blocks[key].append((offset, offset + p.numel()))
        offset += p.numel()
    return {key: ranges for key, ranges in blocks.items() if ranges}


def _flat_scalars(value, prefix=''):
    out = {}
    if isinstance(value, dict):
        for name, child in value.items():
            out.update(_flat_scalars(child, f'{prefix}/{name}' if prefix else str(name)))
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value):
        out[prefix] = float(value)
    return out


def _array_fingerprint(value):
    """Fingerprint diagnostic draws, not checkpoint provenance or model inputs."""
    digest = hashlib.blake2b(digest_size=16)
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        digest.update(json.dumps([str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    else:
        array = np.asarray(value)
        digest.update(json.dumps([str(array.dtype), list(array.shape)]).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _vector_cosine(left, right):
    # Bound temporary FP64 storage even when retaining several full directions.
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError('Direction layouts differ')
    cross, ll, rr = 0., 0., 0.
    for start in range(0, left.numel(), 1_000_000):
        a, b = left[start:start+1_000_000].double(), right[start:start+1_000_000].double()
        cross += float(torch.dot(a, b)); ll += float(torch.dot(a, a)); rr += float(torch.dot(b, b))
    return None if ll == 0 or rr == 0 else float(np.clip(cross / np.sqrt(ll*rr), -1, 1))


def displacement_measurements(requested, realized):
    requested_rms = float(requested.double().square().mean().sqrt())
    realized_rms = float(realized.double().square().mean().sqrt())
    cosine = _vector_cosine(requested, realized)
    ratio = None if requested_rms == 0 else realized_rms / requested_rms
    reasons = []
    if realized_rms == 0:
        reasons.append('parameter displacement rounded to zero')
    if cosine is None or cosine < .95:
        reasons.append('realized/requested cosine below 0.95')
    if ratio is None or not .8 <= ratio <= 1.2:
        reasons.append('realized/requested norm ratio outside [0.8,1.2]')
    return dict(requested_parameter_update_rms=requested_rms,
        parameter_update_rms=realized_rms, realized_requested_cosine=cosine,
        realized_requested_norm_ratio=ratio,
        parameter_changed_fraction=float(realized.ne(0).double().mean()),
        perturbation_valid=not reasons, perturbation_invalid_reasons=reasons)


def generated_motion(before, after, panel):
    """Resolved all-K motion; tau directions use the existing fixed-energy map."""
    a, b = [np.asarray(value['deltas'], dtype=np.float64) for value in (before, after)]
    if a.shape != b.shape or a.ndim != 4 or a.shape[-2:] != (2, 2) or not all(
            np.isfinite(value).all() for value in (a, b)):
        raise ValueError('Finite paired (events,K,2,2) deltas required for motion')
    difference = b-a
    wrapped_phi = np.arctan2(np.sin(difference[..., 1]), np.cos(difference[..., 1]))
    visible = np.stack([panel['visible_a'][:, 1:], panel['visible_b'][:, 1:]], axis=1).astype(np.float64)
    if visible.shape != (len(a), 2, 3) or not np.isfinite(visible).all() or (np.linalg.norm(visible, axis=-1) <= 0).any():
        raise ValueError('Physical visible momenta required for tau direction motion')
    theta = np.arctan2(np.linalg.norm(visible[..., :2], axis=-1), visible[..., 2])[:, None, :]
    phi = np.arctan2(visible[..., 1], visible[..., 0])[:, None, :]
    def direction(delta):
        t, p = theta+delta[..., 0], phi+delta[..., 1]
        return np.stack([np.sin(t)*np.cos(p), np.sin(t)*np.sin(p), np.cos(t)], axis=-1)
    va, vb = direction(a), direction(b)
    angular = np.arctan2(np.linalg.norm(np.cross(va, vb), axis=-1), (va*vb).sum(-1))
    return dict(delta_rms=float(np.sqrt(np.mean(difference**2))),
        delta_theta_rms_rad=float(np.sqrt(np.mean(difference[..., 0]**2))),
        delta_phi_wrapped_rms_rad=float(np.sqrt(np.mean(wrapped_phi**2))),
        candidate_delta_changed_fraction=float(np.any(difference != 0, axis=(-2, -1)).mean()),
        tau_direction_angular_rms_rad=float(np.sqrt(np.mean(angular**2))),
        tau_momentum_rms_fixed_energy=float(np.sqrt((45.6**2-1.777**2)*np.mean(np.sum((vb-va)**2, axis=-1)))),
        motion_resolved=bool(np.any(difference != 0)),
        scope='Unweighted all-event/all-K motion; existing fixed-energy tau reconstruction, not tau energy unfolding or spin closure')


class TauRewardMechanismDriver:
    def __init__(self, cycle, cfg, *, native_step, optimizer, checkpoint, source_step):
        self.cycle, self.cfg, self.step = cycle, dict(cfg), native_step
        self.optimizer, self.source_step = optimizer, int(source_step)
        if cycle.world != 16 and not cfg.get('allow_cpu_fixture', False):
            raise ValueError('Tau mechanism experiments require all sixteen workers')
        if self.source_step != 1920 or self.source_step != int(cfg['source_step']):
            raise ValueError('Mechanism experiment must inherit exact step1920')
        if cfg.get('reference_recenter') or cfg.get('periodic_refits'):
            raise ValueError('Mechanism diagnosis must not recenter reference or refit periodically')
        if cycle.reward.round_id != cfg['expected_source_reward_round'] or cycle.reward.denominator_step != cfg['expected_source_denominator_step']:
            raise ValueError('Inherited classifier round/denominator differs from pinned source')
        saved = checkpoint['dgpo_optimizer_state_dict']
        current = optimizer.state_dict()
        if len(saved['optimizer']['param_groups']) != len(current['optimizer']['param_groups']):
            raise ValueError('Inherited optimizer groups changed')
        for old, live in zip(saved['optimizer']['param_groups'], current['optimizer']['param_groups'], strict=True):
            for key in ('lr', 'weight_decay', 'betas', 'eps', 'group_name', 'params'):
                if old.get(key) != live.get(key):
                    raise ValueError(f'Inherited optimizer {key} changed')
        if saved['scheduler']['last_epoch'] != current['scheduler']['last_epoch']:
            raise ValueError('Inherited scheduler clock changed')
        if len(cycle.validation['source_ids']) != 119002 and not cfg.get('allow_cpu_fixture', False):
            raise ValueError('Require complete filtered 119002-event validation panel')
        self.output = Path(cfg['output_directory']); self.output.mkdir(parents=True, exist_ok=True)
        self.reference_state = _frozen_state(cycle.reference)
        self.ensemble = None
        directory = Path(cfg['ensemble_directory']) if cfg.get('ensemble_directory') is not None else None
        if cfg['mode'] not in ('ensemble', 'repeat_diagnose') and directory is not None and (directory / 'ensemble.pt').is_file():
            from RL.DGPO_neutrino.tau_classifier_ensemble_probe import TauClassifierEnsemble
            self.ensemble = TauClassifierEnsemble.load(directory, cycle.reward, cycle.device)
        if cfg['reward_arm'] in ('member0', 'ensemble', 'all') and self.ensemble is None:
            raise ValueError('Fresh teacher arms require the same completed ensemble artifact')
        self.trajectory_controller = None
        if cfg.get('full_trajectory'):
            from RL.DGPO_neutrino.tau_full_trajectory import FullTrajectoryController
            if cfg['mode'] != 'trajectory' or cfg['reward_arm'] != 'inherited' or self.ensemble is not None:
                raise ValueError('Full trajectory requires the inherited single reward without an ensemble')
            self.trajectory_controller = FullTrajectoryController(cfg['full_trajectory'])
            self.trajectory_controller.failure_directory = self.output / 'trajectory-failures'
            saved_balance = (checkpoint.get('dgpo_omnifold_reward_stack') or {}).get('trajectory_reference_balance_state')
            if saved_balance is not None:
                if self.trajectory_controller.balance is None:
                    raise ValueError('Checkpoint carries dynamic balance state but the configuration disables it')
                self.trajectory_controller.balance.load_state_dict(saved_balance)
        self.observer = None
        self.replay = None
        if self.ensemble is not None:
            from RL.DGPO_neutrino.tau_native_batch_replay import NativeTrainingReplay
            self.replay = NativeTrainingReplay.load(directory / 'native_training',
                rank=cycle.rank, world=cycle.world, expected_events=416701,
                source_checkpoint=cfg['source_checkpoint'])
        if cfg['mode'] == 'repeat_diagnose':
            if cfg['reward_arm'] != 'inherited' or cfg.get('updates', 0) != 0 or cfg.get('persistent_updates', 0) != 0:
                raise ValueError('Repeated diagnosis requires the unchanged inherited head and zero persistent updates')
            seeds = cfg.get('evaluation_seeds', [])
            radii = cfg.get('direction_rms_fractions', [])
            if (type(cfg.get('draw_repeats')) is not int or cfg['draw_repeats'] < 2
                    or cfg.get('gradient_repeats') != 2 or len(seeds) < 2
                    or len(set(seeds)) != len(seeds) or any(type(seed) is not int or seed < 0 for seed in seeds)
                    or not radii or len(set(radii)) != len(radii)
                    or any(not np.isfinite(radius) or radius <= 0 or radius > 1 for radius in radii)):
                raise ValueError('Declare independent draw repeats, two capture replicas, distinct evaluation seeds and positive radii')
            self.repeat_head_state = _frozen_state(cycle.reward.head)
            self.repeat_reward_clocks = (cycle.reward.round_id, cycle.reward.denominator_step)
            self.repeat_draw = 0
            self.repeat_seen_ids = set()
            self.repeat_baselines = {}
            self.repeat_direction_vectors = {}
            self.repeat_report = dict(complete=False, source_step=self.source_step, actual_updates=0,
                classifier_fits=0, evaluator='installed_training_head', comparable_across_arms=False,
                inherited_reference=True, inherited_adamw=True, source_checkpoint=str(cfg['source_checkpoint']) if cfg.get('source_checkpoint') is not None else None,
                reward_round=cycle.reward.round_id, denominator_step=cycle.reward.denominator_step,
                draw_repeats=cfg['draw_repeats'], gradient_repeats=2, evaluation_seeds=list(seeds),
                direction_rms_fractions=list(radii), native_training_inputs='Fresh nonoverlapping native Ray batches at the same checkpoint',
                primary='Frozen installed-head held-out all-K mean reward change; not independent classifier closure',
                uncertainty='Separately sampled nonoverlapping native batch/candidate draws; two fixed-candidate t/noise replicas within each draw; independent evaluation generation seeds. Event intervals are conditional paired event bootstrap, not pooled update-draw confidence intervals.',
                scope='Read-only finite-radius parameter-space direction diagnostic; raw descent vs inherited AdamW displacement are not function-distance matched. No refit, recenter or persistent update.',
                perturbation_validity='Nonzero realized parameter motion; realized/requested cosine>=0.95 and normratio in[0.8,1.2]; nonzero paired generated delta motion. Invalid measurements are saved, not counted as sign failures.',
                perturbation_threshold_scope='Predeclared engineering resolution/fidelity checks, not physical significance thresholds',
                condition_scope='Legacy opening/noise strata retained only for gradient reconstruction; no condition efficacy inference.',
                draws={}, cross_draw_direction_cosines={})

    def prepare(self, epoch, *, train_shard=None, loader_cfg=None):
        if self.trajectory_controller is not None:
            from RL.DGPO_neutrino.tau_native_batch_replay import ensure_native_training_replay
            self.replay = ensure_native_training_replay(train_shard, loader_cfg,
                self.cfg['native_training_directory'], rank=self.cycle.rank, world=self.cycle.world,
                device=self.cycle.device, seed=int(self.cfg['bootstrap_seed']), expected_events=416701,
                source_checkpoint=self.cfg['source_checkpoint'])
            # Publish actual replay provenance, including for the first launch
            # whose CLI preflight correctly observed that it did not exist yet.
            metadata_keys = ('kind', 'schema_version', 'source_policy_step', 'source_checkpoint', 'world',
                'global_rows', 'batch_size', 'base_shuffle_seed', 'global_replay_fingerprint', 'tail_recipe')
            publication_error = [None]
            if self.cycle.rank == 0:
                try:
                    manifest_path = self.output.parent / 'invocation_manifest.json'
                    manifest = json.loads(manifest_path.read_text())
                    manifest['native_training_replay'] = {key: self.replay.manifest[key] for key in metadata_keys}
                    pending = manifest_path.with_suffix('.pending.json')
                    pending.write_text(json.dumps(manifest, indent=2)+'\n')
                    pending.replace(manifest_path)
                except Exception as exc:
                    publication_error[0] = str(exc)
            if self.cycle.world > 1:
                dist.broadcast_object_list(publication_error, src=0)
            if publication_error[0] is not None:
                raise ValueError(f'Cannot publish replay provenance: {publication_error[0]}')
        if self.cfg['mode'] == 'ensemble':
            from RL.DGPO_neutrino.tau_classifier_ensemble_probe import fit_tau_ensemble, score_saved_ensemble_panel
            from RL.DGPO_neutrino.checkpoint_transfer import isolated_evaluation
            folder = Path(self.cfg['ensemble_directory'])
            self.cycle.train = self.cycle.read_panel(self.cycle.cfg['train_panel'])
            if len(self.cycle.train['source_ids']) != 416701 or np.intersect1d(
                    self.cycle.train['source_ids'], self.cycle.validation['source_ids']).size:
                raise ValueError('Require complete disjoint filtered train/validation panels')
            with isolated_evaluation(self.cycle.actor, int(self.cycle.cfg['refit_seed']) + self.cycle.rank):
                generated = self.cycle.generate(self.cycle.train, folder / 'matched-train', 1,
                    self.cycle.cfg['refit_seed'], False)
            self.ensemble = fit_tau_ensemble(self.cycle, generated, folder, step=self.source_step,
                seeds=self.cfg['ensemble_fit_seeds'], judge_seed=self.cfg['ensemble_judge_seed'],
                source_checkpoint=self.cfg['source_checkpoint'])
            _, evaluated = self._evaluate('ensemble-validation')
            saved_validation = self.cycle.validation
            try:
                self.cycle.validation = self._panel()
                report = score_saved_ensemble_panel(self.cycle, self.ensemble, evaluated,
                    self.output / 'ensemble-validation' / 'candidates', self.output / 'ensemble-scores', step=self.source_step)
            finally:
                self.cycle.validation = saved_validation
            from RL.DGPO_neutrino.tau_native_batch_replay import record_native_training_replay
            record_native_training_replay(train_shard, loader_cfg, folder / 'native_training',
                rank=self.cycle.rank, world=self.cycle.world, device=self.cycle.device,
                seed=int(self.cfg['bootstrap_seed']), expected_events=416701,
                source_checkpoint=self.cfg['source_checkpoint'])
            if report is not None:
                report.update(complete=True, native_training_directory=str(folder / 'native_training'),
                    native_sampling_seed=int(self.cfg['bootstrap_seed']))
            self._write('ensemble_diagnostic.json', report)
            self._assert_reference()
            return True
        if self.cfg['mode'] == 'trajectory':
            fresh_refit = (self.cfg.get('full_trajectory') or {}).get('reward_refit')
            if fresh_refit is not None:
                from RL.DGPO_neutrino.tau_trajectory_reward_refit import prepare_trajectory_reward
                self.reward_refit_report = prepare_trajectory_reward(self.cycle, fresh_refit,
                    self.output / 'startup-reward-refit', epoch=epoch, step=self.source_step,
                    min_selected_steps=1000 if self.trajectory_controller.classifier_kl_cfg is not None else 0)
                # The explicit startup transaction establishes the new fixed
                # anchor; subsequent updates may never recenter it again.
                self.reference_state = _frozen_state(self.cycle.reference)
            if self.trajectory_controller is not None and self.trajectory_controller.classifier_kl_cfg is not None:
                from RL.DGPO_neutrino.tau_classifier_kl import TauClassifierKL
                self.trajectory_controller.classifier_kl = TauClassifierKL(self.cycle,
                    self.trajectory_controller.classifier_kl_cfg, self.output / 'classifier-kl',
                    anchor_directory=self.output / 'startup-reward-refit' / 'anchor-candidates')
                if self.trajectory_controller.hard_trust_cfg is not None:
                    from RL.DGPO_neutrino.tau_classifier_trust import TauClassifierTrust
                    self.trajectory_controller.hard_trust = TauClassifierTrust(
                        self.trajectory_controller.classifier_kl, self.trajectory_controller.hard_trust_cfg,
                        self.output / 'classifier-trust')
            if self.cfg['reward_arm'] != 'inherited':
                selection = 'single' if self.cfg['reward_arm'] == 'member0' else self.cfg['reward_arm']
                self.ensemble.install_into_reward(self.cycle.reward, selection,
                    policy_step=self.source_step, epoch=epoch)
            from RL.DGPO_neutrino.tau_reward_transfer import TauRewardTransferProbe
            options = dict(self.cfg, mechanism_protocol=True, gradient_trace_enabled=False,
                marginal_relative_steps=[0, self.cfg['updates']])
            if self.ensemble is None:
                options['ensemble_directory'] = None
            self.observer = TauRewardTransferProbe(self.cycle, options, source_step=self.source_step,
                reward_round=self.cycle.reward.round_id, ensemble=self.ensemble)
            self.observer.start()
        if self.cfg['mode'] == 'repeat_diagnose':
            self._assert_repeat_frozen()
            self._write('local_direction_repeats.json', self.repeat_report)
        self._assert_reference()
        return False

    def training_iterator(self, shard, loader_cfg):
        return self.replay.iterator() if self.replay is not None else iter(shard.iter_torch_batches(**loader_cfg))

    def _assert_reference(self):
        if _frozen_state(self.cycle.reference) != self.reference_state:
            raise ValueError('Original velocity reference changed during mechanism experiment')

    def _write(self, name, value):
        if self.cycle.rank == 0:
            if value is None:
                raise RuntimeError('Missing rank0 diagnostic report')
            path = self.output / name
            temporary = path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
            temporary.replace(path)
            if name != 'local_direction_repeats.json':
                self.cycle.emit({f'tau/mechanism/{name.removesuffix(".json")}/{key}': val
                                 for key, val in _flat_scalars(value).items()}, self.source_step)
        if self.cycle.world > 1:
            dist.barrier()

    def _panel(self):
        panel = self.cycle.validation
        count = int(self.cfg['evaluation_events'])
        if count > len(panel['source_ids']) or count < 2:
            raise ValueError('Invalid independent evaluation event count')
        return {key: value[:count] if isinstance(value, np.ndarray) and value.ndim and
                len(value) == len(panel['source_ids']) else value for key, value in panel.items()}

    def _evaluate(self, name, seed=None):
        from RL.DGPO_neutrino.checkpoint_transfer import isolated_evaluation
        panel = self._panel(); folder = self.output / name
        seed = int(self.cycle.cfg['validation_seed']) if seed is None else int(seed)
        with isolated_evaluation(self.cycle.actor, seed + self.cycle.rank):
            generated = self.cycle.generate(panel, folder / 'candidates', 8,
                seed, True, **({'retain_features': False} if self.cfg['mode'] == 'repeat_diagnose' else {}))
        training = generated.get('rewards')
        if self.ensemble is not None:
            # Clone the physics dict: the scorer replaces rewards only on rank0.
            judge = dict(generated)
            if 'rewards' in generated:
                judge['rewards'] = generated['rewards'].copy()
            saved_validation = self.cycle.validation
            try:
                self.cycle.validation = panel
                with self.ensemble.temporary_reward(self.cycle.reward, 'judge'):
                    self.cycle._score_transfer_head(judge, folder / 'candidates', folder / 'judge',
                        step=self.source_step, phase='mechanism_judge')
            finally:
                self.cycle.validation = saved_validation
            generated['rewards'] = judge.get('rewards')
        if self.cycle.rank == 0:
            from RL.DGPO_neutrino.conditional_tau_cycle import cij_terms
            generated['training_reward'] = training
            generated['truth_cij'] = cij_terms(panel, np.arange(len(panel['source_ids'])), panel['truth_deltas'])
            evaluation_arrays = {'judge_reward': generated['rewards']} if self.ensemble is not None else {}
            np.savez_compressed(folder / 'measurements.npz', source_ids=panel['source_ids'],
                reward=generated['rewards'], training_reward=training, generated_cij=generated['cij'],
                truth_cij=generated['truth_cij'], weight=panel['event_weight'], deltas=generated['deltas'],
                **evaluation_arrays)
        self._assert_reference()
        return panel, generated

    def _compare(self, baseline, generated, panel):
        if self.cycle.rank != 0:
            return None
        options = dict(replicates=int(self.cfg['bootstrap_replicates']), seed=int(self.cfg['bootstrap_seed']))
        report = paired_statistics(baseline['rewards'], generated['rewards'], baseline['cij'], generated['cij'],
            generated['truth_cij'], panel['event_weight'], **options)
        report['training_reward'] = reward_statistics(generated['training_reward'], panel['event_weight'])
        if self.cfg['mode'] == 'repeat_diagnose':
            return report
        va, vb = panel['visible_a'][:, 1:], panel['visible_b'][:, 1:]
        if (not np.isfinite(va).all() or not np.isfinite(vb).all()
                or (np.linalg.norm(va, axis=1) <= 0).any()
                or (np.linalg.norm(vb, axis=1) <= 0).any()):
            raise ValueError('Condition strata require finite nonzero observed visible momenta')
        cosine = (va*vb).sum(1) / (np.linalg.norm(va, axis=1)*np.linalg.norm(vb, axis=1))
        labels = np.searchsorted(self.cfg['condition_edges'][1:-1], np.clip(cosine, -1, 1), side='right')
        report['condition_transfer'] = {}
        for index in range(int(self.cfg['condition_groups'])):
            rows = labels == index
            if rows.sum() >= 2 and panel['event_weight'][rows].sum() > 0:
                report['condition_transfer'][str(index)] = paired_statistics(
                    baseline['rewards'][rows], generated['rewards'][rows], baseline['cij'][rows],
                    generated['cij'][rows], generated['truth_cij'][rows], panel['event_weight'][rows], **options)
        return report

    def _assert_repeat_frozen(self):
        self._assert_reference()
        if (_frozen_state(self.cycle.reward.head) != self.repeat_head_state or
                (self.cycle.reward.round_id, self.cycle.reward.denominator_step) != self.repeat_reward_clocks):
            raise ValueError('Inherited classifier or reward clocks changed during repeated diagnosis')

    def _emit_repeat_draw(self, draw, *, done):
        """Stable small metric vocabulary; detailed indexed data stays in JSON."""
        if self.cycle.rank != 0:
            return
        prefix = 'tau/local_direction'
        current = self.repeat_report['draws'][str(draw)]
        payload = {f'{prefix}/draw_index': float(draw), f'{prefix}/complete': float(done),
                   f'{prefix}/persistent_updates': 0., f'{prefix}/completed_draws': float(draw+1),
                   f'{prefix}/native_parameter_update_rms': current['native_parameter_update_rms']}
        payload[f'{prefix}/reconstruction_relative_error_max'] = max(
            value['reconstruction']['relative_error'] for value in current['gradient_replicas'].values())
        if current.get('fixed_candidate_total_gradient_cosine') is not None:
            payload[f'{prefix}/fixed_candidate_noise_cosine_total'] = current['fixed_candidate_total_gradient_cosine']
        for component in ('reward', 'reference'):
            cosine = current['sampling_stability']['components'][component]['total']['mean_pairwise_cosine']
            if cosine is not None:
                payload[f'{prefix}/fixed_candidate_noise_cosine_{component}'] = cosine
        for direction in ('raw_reward', 'raw_total', 'native_adamw'):
            matrix_row = self.repeat_report['cross_draw_direction_cosines'][direction][str(draw)]
            if matrix_row.get('0') is not None:
                payload[f'{prefix}/{direction}/cosine_to_draw0'] = matrix_row['0']
            previous = [value for other, value in matrix_row.items() if int(other) < draw and value is not None]
            if previous:
                payload[f'{prefix}/{direction}/min_previous_draw_cosine'] = min(previous)
            for fraction in self.cfg['direction_rms_fractions']:
                rows = [current['interventions'][f'{direction}_rms{fraction:g}_sign{sign:+d}']['evaluations'] for sign in (1, -1)]
                plus, minus = [np.array([values[str(seed)]['reward']['delta_mean'] for seed in self.cfg['evaluation_seeds']]) for values in rows]
                stem = f'{prefix}/{direction}/rms{fraction:g}'
                payload.update({f'{stem}/plus_mean_gain': float(plus.mean()),
                    f'{stem}/minus_mean_gain': float(minus.mean()), f'{stem}/evaluation_seed_spread': float(np.ptp(plus)),
                    f'{stem}/odd_response': float(((plus-minus)/2).mean()),
                    f'{stem}/even_response': float(((plus+minus)/2).mean()),
                    f'{stem}/measurement_valid': float(all(value['measurement_valid'] for values in rows for value in values.values())),
                    f'{stem}/tau_direction_angular_rms_rad': float(np.mean([value['motion']['tau_direction_angular_rms_rad'] for value in rows[0].values()])),
                    f'{stem}/realized_parameter_update_rms': float(np.mean([value['motion']['parameter_update_rms'] for value in rows[0].values()]))})
                for seed in self.cfg['evaluation_seeds']:
                    reward = rows[0][str(seed)]['reward']
                    payload[f'{stem}/seed{seed}/plus_gain'] = reward['delta_mean']
                    payload[f'{stem}/seed{seed}/plus_lo95'] = reward['delta_lo95']
                    payload[f'{stem}/seed{seed}/plus_hi95'] = reward['delta_hi95']
        self.cycle.emit(payload, self.source_step)

    def _gather_records(self, record):
        if self.cycle.world == 1:
            return [record]
        records = [None] * self.cycle.world
        dist.all_gather_object(records, record)
        return records

    def _repeat_batch_identity(self, batch):
        record = dict(rank=self.cycle.rank, error=None)
        try:
            from scripts.diagnose_ztautau_cij import SOURCE_ID_COLUMNS, source_ids
            from RL.DGPO_neutrino.tau_native_batch_replay import batch_fingerprint
            columns = {key: (value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value))
                       for key in SOURCE_ID_COLUMNS if (value := batch.get(key)) is not None}
            schema = tuple(columns)
            # Encoded panels do not name individual optional components. A gap
            # would make a three-component ID ambiguous (file vs event index).
            if schema != SOURCE_ID_COLUMNS[:len(schema)]:
                raise ValueError(f'Nonprefix source identity schema {schema}; do not infer missing optional fields')
            identities = source_ids(columns, schema).tolist()
            base_ids = source_ids(columns, SOURCE_ID_COLUMNS[:2]).tolist()
            if len(identities) != len(batch['x']):
                raise ValueError(f'Native source identity count {len(identities)} differs from batch count {len(batch["x"])}')
            unique, counts = np.unique(identities, return_counts=True)
            if (counts > 1).any():
                raise ValueError(f'Duplicate full source identities within a native rank: columns={schema}, '
                                 f'duplicate_groups={int((counts > 1).sum())}, examples={unique[counts > 1][:5].tolist()}')
            if not self.cfg.get('allow_cpu_fixture', False) and len(identities) != int(self.cfg['gradient_events_per_rank']):
                raise ValueError('Native draw event count differs from predeclared gradient batch')
            # Report rank-local panel failures collectively before any worker
            # can enter gradient collectives or commit the seen-identity set.
            validation_values = np.asarray(self.cycle.validation['source_ids'])
            if validation_values.ndim != 1 or not len(validation_values):
                raise ValueError('Held-out panel requires a nonempty source identity vector')
            validation_ids = validation_values.astype(str).tolist()
            widths = {len(identity.split(':')) for identity in validation_ids}
            if widths != {len(schema)}:
                raise ValueError(f'Held-out source identity width {sorted(widths)} differs from native columns {schema}; no truncation allowed')
            record.update(event_count=len(identities), source_id_columns=list(schema),
                          source_ids=identities, base_source_ids=base_ids,
                          base_id_collision_count=len(base_ids)-len(set(base_ids)),
                          event_id_fingerprint=_array_fingerprint(np.asarray(identities)), batch_fingerprint=batch_fingerprint(batch),
                          heldout_event_count=len(validation_ids),
                          heldout_event_id_fingerprint=_array_fingerprint(np.asarray(validation_ids)))
        except Exception as error:
            record['error'] = f'{type(error).__name__}: {error}'
        records = self._gather_records(record)
        if any(item['error'] for item in records):
            raise ValueError(f'Invalid native draw identities: {[item["error"] for item in records]}')
        schemas = {tuple(item['source_id_columns']) for item in records}
        if len(schemas) != 1:
            raise ValueError(f'Native source identity schema differs across ranks: {sorted(schemas)}')
        schema = next(iter(schemas))
        if getattr(self, 'repeat_source_id_columns', schema) != schema:
            raise ValueError('Native source identity schema changed across diagnostic draws')
        heldout_views = {(item['heldout_event_count'], item['heldout_event_id_fingerprint']) for item in records}
        if len(heldout_views) != 1:
            raise ValueError('Held-out source identity views differ across native ranks')
        identities = [identity for item in records for identity in item['source_ids']]
        unique, counts = np.unique(identities, return_counts=True)
        if (counts > 1).any():
            raise ValueError(f'Duplicate full source identities across native ranks: columns={schema}, '
                             f'duplicate_groups={int((counts > 1).sum())}, examples={unique[counts > 1][:5].tolist()}')
        seen_overlap = set(identities) & self.repeat_seen_ids
        if seen_overlap:
            raise ValueError(f'Repeated direction diagnosis requires nonoverlapping native batches: '
                             f'{len(seen_overlap)} repeated full source identities, examples={sorted(seen_overlap)[:5]}')
        validation_overlap = set(identities) & set(validation_ids)
        if validation_overlap:
            raise ValueError(f'Native gradient full source identities overlap held-out validation identities: '
                             f'count={len(validation_overlap)}, examples={sorted(validation_overlap)[:5]}')
        self.repeat_source_id_columns = schema
        self.repeat_seen_ids.update(identities)
        base_ids = [identity for item in records for identity in item['base_source_ids']]
        return dict(global_event_count=len(identities), global_unique_event_count=len(set(identities)),
                    source_id_columns=list(schema), base_id_collision_count=len(base_ids)-len(set(base_ids)),
                    event_id_fingerprint=_array_fingerprint(np.asarray(identities)), ranks={str(item['rank']): item for item in records})

    def _repeat_anchor_baselines(self):
        if self.repeat_baselines:
            return
        for seed in self.cfg['evaluation_seeds']:
            panel, baseline = self._evaluate(f'repeat/baseline/seed{seed}', seed=seed)
            _, replay = self._evaluate(f'repeat/zero-replay/seed{seed}', seed=seed)
            error = max(float(np.max(np.abs(baseline[key]-replay[key]))) for key in ('rewards', 'deltas')) if self.cycle.rank == 0 else 0.
            check = torch.tensor(error, device=self.cycle.device, dtype=torch.float64)
            if self.cycle.world > 1:
                dist.all_reduce(check, op=dist.ReduceOp.MAX)
            if not torch.isfinite(check) or float(check) > 1e-6:
                raise ValueError('Repeated diagnosis zero-update CRN replay failed')
            # Features are large and unnecessary for paired motion/reward stats.
            saved = {key: baseline[key] for key in ('rewards', 'training_reward', 'cij', 'truth_cij', 'deltas') if key in baseline}
            self.repeat_baselines[str(seed)] = (panel, saved)
            if self.cycle.rank == 0:
                self.repeat_report.setdefault('anchor_baselines', {})[str(seed)] = dict(
                    generation_seed=seed, zero_replay_max_abs_error=float(check), event_count=len(panel['source_ids']),
                    event_id_fingerprint=_array_fingerprint(panel['source_ids']),
                    candidate_delta_fingerprint=_array_fingerprint(baseline['deltas']))
            del baseline, replay

    def _repeat_native_step(self, *args, **kwargs):
        if int(kwargs['global_step']) != self.source_step or self.repeat_draw >= self.cfg['draw_repeats']:
            raise ValueError('Repeated diagnosis must stay at source step until all draws complete')
        self._assert_repeat_frozen()
        actor, batch, optimizer = args[0], args[4], args[5]
        parameters = tuple(p for p in actor.parameters() if p.requires_grad)
        draw = self.repeat_draw
        identity = self._repeat_batch_identity(batch)
        self._repeat_anchor_baselines()
        captures, summaries, vectors, presence, candidates = [], {}, [], [], None
        draw_seed = int(self.cfg['bootstrap_seed']) + draw * 1_000_003
        candidate_record = dict(rank=self.cycle.rank, candidate_seed=draw_seed+self.cycle.rank, replica_seeds=[], replica_fingerprints={})
        for repeat in range(2):
            trace = NativeGradientAccumulator(parameters=parameters,
                condition_names=['visible_opposed', 'visible_middle', 'visible_aligned'], noise_band_names=['low_t', 'mid_t', 'high_t'])
            trace.condition_edges, trace.noise_edges = self.cfg['condition_edges'], self.cfg['noise_edges']
            seed = draw_seed + self.cycle.rank + repeat*100_003
            with native_transaction(actor, optimizer, seed=seed):
                result = self.step(*args, **dict(kwargs, mechanism_capture=trace,
                    mechanism_read_only=True, mechanism_candidates=candidates))
            candidates = result['mechanism_candidates']
            candidate_record['replica_seeds'].append(seed)
            candidate_record['replica_fingerprints'][str(repeat)] = _array_fingerprint(candidates)
            summary = trace.summarize(actual_total_gradient=result['mechanism_vectors']['actual_unclipped'],
                trust_coefficient=kwargs['reference_trust_coefficient'], parameter_blocks=parameter_blocks(actor),
                reconstruction_tolerance=float(self.cfg['reconstruction_tolerance']))
            if not summary['reconstruction']['passed']:
                raise ValueError('Repeated native gradient reconstruction failed')
            captures.append(trace); summaries[str(repeat)] = summary
            vectors.append(result['mechanism_vectors']['actual_unclipped'])
            presence.append(result['mechanism_gradient_present'])
        candidate_records = self._gather_records(candidate_record)
        if any(item['replica_fingerprints']['0'] != item['replica_fingerprints']['1'] for item in candidate_records):
            raise ValueError('Capture replicas must reuse exactly the same native candidates within each draw')
        native = adamw_proposal(actor, optimizer, parameters, vectors[0], presence[0],
            clip_norm=float(kwargs['grad_clip_norm']), seed=draw_seed+self.cycle.rank)
        rms = float(native.double().square().mean().sqrt())
        # Match the same candidate/t/eps/dropout draw to the native AdamW
        # proposal. Replica1 is a stability check, not an averaging treatment.
        raw = dict(raw_reward=captures[0].total('reward'), raw_total=vectors[0])
        directions = {name: normalized_descent(value, rms) if rms > 0 else torch.zeros_like(value) for name, value in raw.items()}
        directions['native_adamw'] = native
        directions = {name: torch.zeros_like(native) if value is None else value for name, value in directions.items()}
        stability = summarize_replicas(captures)
        fixed_candidate_total_cosine = _vector_cosine(vectors[0], vectors[1])
        del captures, vectors, presence, candidates, result, trace, raw
        self.repeat_direction_vectors[str(draw)] = directions
        if self.cycle.rank == 0:
            self.repeat_report['draws'][str(draw)] = dict(native_batch=identity,
                candidates={str(item['rank']): item for item in candidate_records}, gradient_replicas=summaries,
                sampling_stability=stability, native_parameter_update_rms=rms,
                fixed_candidate_total_gradient_cosine=fixed_candidate_total_cosine,
                native_proposal_gradient_replica=0, direction_gradient_replica=0,
                raw_direction_gradient='replica0, same native candidates/t/eps/dropout as inherited AdamW proposal; replica1 only checks stability',
                interventions={})
            for direction in directions:
                matrix = self.repeat_report['cross_draw_direction_cosines'].setdefault(direction, {})
                for left, values in self.repeat_direction_vectors.items():
                    cosine = _vector_cosine(values[direction], directions[direction])
                    matrix.setdefault(left, {})[str(draw)] = cosine
                    matrix.setdefault(str(draw), {})[left] = cosine
        self._write('local_direction_repeats.json', self.repeat_report)
        for direction, displacement in directions.items():
            for fraction in self.cfg['direction_rms_fractions']:
                for sign in (1, -1):
                    key = f'{direction}_rms{fraction:g}_sign{sign:+d}'
                    evaluations = {}
                    requested = displacement * (fraction*sign)
                    for seed in self.cfg['evaluation_seeds']:
                        panel, baseline = self.repeat_baselines[str(seed)]
                        with native_transaction(actor, optimizer, seed=draw_seed+self.cycle.rank):
                            before = parameter_vector(parameters)
                            add_displacement(parameters, requested)
                            realized = parameter_vector(parameters)-before
                            parameter_motion = displacement_measurements(requested, realized)
                            _, generated = self._evaluate(f'repeat/draw{draw}/{key}/seed{seed}', seed=seed)
                        effect = self._compare(baseline, generated, panel)
                        if effect is not None:
                            motion = dict(parameter_motion, **generated_motion(baseline, generated, panel))
                            reasons = list(motion['perturbation_invalid_reasons'])
                            if not motion['motion_resolved']:
                                reasons.append('paired generated delta motion rounded to zero')
                            effect.update(generation_seed=seed, parameter_update_rms=motion['parameter_update_rms'],
                                motion=motion, measurement_valid=not reasons, measurement_invalid_reasons=reasons)
                            evaluations[str(seed)] = effect
                        del generated, before, realized
                    if self.cycle.rank == 0:
                        self.repeat_report['draws'][str(draw)]['interventions'][key] = dict(
                            direction=direction, fraction=float(fraction), sign=sign, evaluations=evaluations)
                    self._assert_repeat_frozen()
                    self._write('local_direction_repeats.json', self.repeat_report)
        self.repeat_draw += 1
        done = self.repeat_draw == self.cfg['draw_repeats']
        if self.cycle.rank == 0:
            from RL.DGPO_neutrino.tau_direction_repeat_statistics import summarize_direction_repeats
            self.repeat_report['summary'] = summarize_direction_repeats(self.repeat_report['draws'])
            self.repeat_report['completed_draws'] = self.repeat_draw
            self.repeat_report['complete'] = done
        self._assert_repeat_frozen()
        self._write('local_direction_repeats.json', self.repeat_report)
        self._emit_repeat_draw(draw, done=done)
        if done:
            self.repeat_direction_vectors.clear()
            self.repeat_baselines.clear()
        return {'_tau_mechanism_diagnosis_complete': True, 'train/optimizer_step_ran': 0.0} if done else {
            '_tau_mechanism_read_only_pending': True, 'train/optimizer_step_ran': 0.0}

    def native_step(self, *args, **kwargs):
        if self.cfg['mode'] == 'repeat_diagnose':
            return self._repeat_native_step(*args, **kwargs)
        if self.cfg['mode'] != 'diagnose':
            seed = int(self.cfg['bootstrap_seed']) + self.cycle.rank + (int(kwargs['global_step'])-self.source_step)*100003
            if self.trajectory_controller is not None:
                kwargs['trajectory_controller'] = self.trajectory_controller
                if self.trajectory_controller.classifier_kl is not None:
                    self._assert_reference()
                    self.trajectory_controller.classifier_kl.refresh(int(kwargs['global_step']))
                    if kwargs.get('reference_trust_coefficient', 0) != 0:
                        raise ValueError('Classifier KL must disable native velocity penalty')
            with matched_sampling(seed, device=self.cycle.device):
                return self.step(*args, **kwargs)
        actor, optimizer = args[0], args[5]
        parameters = tuple(p for p in actor.parameters() if p.requires_grad)
        arms = ['inherited', 'member0', 'ensemble'] if self.cfg['reward_arm'] == 'all' else [self.cfg['reward_arm']]
        report = dict(complete=False, source_step=self.source_step, actual_updates=0,
            inherited_reference=True, inherited_adamw=True,
            evaluator='independent_fifth_fixed_head' if self.ensemble else 'installed_training_head',
            comparable_across_arms=self.ensemble is not None,
            native_training_inputs='shared recorded native batches' if self.replay else 'live Ray batch, not paired across separate launches',
            scope='Local native gradients and held-out signed parameter-RMS perturbations; NOT function-distance matched. Original RNG/data iterator not checkpointed.',
            condition_scope='Observed lab visible opening strata; not spin variables or a test of all conditioning information.',
            arms={})
        for arm in arms:
            head_context = nullcontext() if arm == 'inherited' else self.ensemble.temporary_reward(self.cycle.reward, arm)
            with head_context:
                panel, baseline = self._evaluate(f'{arm}/baseline')
                _, replay = self._evaluate(f'{arm}/zero-replay')
                error = max(float(np.max(np.abs(baseline[k]-replay[k]))) for k in ('rewards', 'deltas')) if self.cycle.rank == 0 else 0.
                check = torch.tensor(error, device=self.cycle.device, dtype=torch.float64)
                if self.cycle.world > 1:
                    dist.all_reduce(check, op=dist.ReduceOp.MAX)
                if not torch.isfinite(check) or float(check) > 1e-6:
                    raise ValueError('Zero-update CRN replay failed; do not interpret interventions')
                captures, replicas, vectors, presence_masks, candidates = [], [], [], [], None
                for repeat in range(int(self.cfg['gradient_repeats'])):
                    trace = NativeGradientAccumulator(parameters=parameters,
                        condition_names=['visible_opposed', 'visible_middle', 'visible_aligned'],
                        noise_band_names=['low_t', 'mid_t', 'high_t'])
                    trace.condition_edges, trace.noise_edges = self.cfg['condition_edges'], self.cfg['noise_edges']
                    seed = int(self.cfg['bootstrap_seed']) + self.cycle.rank + repeat*100003
                    with native_transaction(actor, optimizer, seed=seed):
                        result = self.step(*args, **dict(kwargs, mechanism_capture=trace,
                            mechanism_read_only=True, mechanism_candidates=candidates))
                    candidates = result['mechanism_candidates']
                    summary = trace.summarize(actual_total_gradient=result['mechanism_vectors']['actual_unclipped'],
                        trust_coefficient=kwargs['reference_trust_coefficient'],
                        parameter_blocks=parameter_blocks(actor),
                        reconstruction_tolerance=float(self.cfg['reconstruction_tolerance']))
                    if not summary['reconstruction']['passed']:
                        raise ValueError('Native gradient reconstruction failed; abort direction experiments')
                    captures.append(trace); replicas.append(summary); vectors.append(result['mechanism_vectors']['actual_unclipped'])
                    presence_masks.append(result['mechanism_gradient_present'])
                full = torch.stack(vectors).mean(0)
                reward = torch.stack([trace.total('reward') for trace in captures]).mean(0)
                reference = torch.stack([trace.total('reference') for trace in captures]).mean(0)
                seed = int(self.cfg['bootstrap_seed']) + self.cycle.rank
                # Replica0 is an actual native draw. The two-replica mean is
                # reserved for geometry directions, not called a native step.
                native = adamw_proposal(actor, optimizer, parameters, vectors[0], presence_masks[0],
                    clip_norm=float(kwargs['grad_clip_norm']), seed=seed)
                no_ref = adamw_proposal(actor, optimizer, parameters, captures[0].total('reward'), presence_masks[0],
                    clip_norm=float(kwargs['grad_clip_norm']), seed=seed)
                rms = float(native.double().square().mean().sqrt())
                directions = {'full': full, 'reward': reward, 'reference': reference}
                for condition in captures[0].condition_names:
                    directions[f'condition_{condition}'] = torch.stack([
                        sum((tr.cells[condition, noise]['reward'] for noise in tr.noise_band_names), torch.zeros_like(full))
                        for tr in captures]).mean(0)
                for noise in captures[0].noise_band_names:
                    directions[f'noise_{noise}'] = torch.stack([
                        sum((tr.cells[condition, noise]['reward'] for condition in tr.condition_names), torch.zeros_like(full))
                        for tr in captures]).mean(0)
                effects = {}
                def measure(name, displacement):
                    with native_transaction(actor, optimizer, seed=seed):
                        add_displacement(parameters, displacement)
                        pp, generated = self._evaluate(f'{arm}/{name}')
                    effect = self._compare(baseline, generated, pp)
                    if effect is not None:
                        effect['parameter_update_rms'] = float(displacement.double().square().mean().sqrt())
                        effects[name] = effect
                measure('native_adamw', native)
                measure('adamw_without_reference_gradient', no_ref)
                if rms > 0:
                    for name, gradient in directions.items():
                        delta = normalized_descent(gradient, rms)
                        if delta is not None:
                            for fraction in self.cfg.get('direction_rms_fractions', [1.0]):
                                for sign in (1, -1):
                                    measure(f'{name}_rms{fraction:g}_sign{sign:+d}', delta * fraction * sign)
                if self.cycle.rank == 0:
                    report['arms'][arm] = dict(zero_replay_max_abs_error=float(check),
                        gradient_replicas=replicas, sampling_stability=summarize_replicas(captures),
                        native_parameter_update_rms=rms, native_proposal_gradient_replica=0,
                        signed_direction_gradient='mean of two fixed-candidate gradient sampling replicas',
                        interventions=effects)
                self._write('gradient_diagnosis.json', report)
                del captures, replicas, vectors
        report['complete'] = True
        self._write('gradient_diagnosis.json', report)
        return {'_tau_mechanism_diagnosis_complete': True, 'train/optimizer_step_ran': 0.0}
