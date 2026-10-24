"""Current-policy tau refits and independent Cij/audit monitoring on all ranks.

Runs inside the existing native DGPO worker group: no nested Ray trainer and no
additional GPU reservation. Refits replace the reward; only an accepted best-val
head install recenters the velocity reference. Actor AdamW/clock are untouched.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F

from RL.DGPO_neutrino.conditional_tau_reward import (
    feature_precision, canonical_policy_batch, POLICY_CONDITIONING_CONTRACT,
    SUPERVISED_POLICY_CONDITIONING_CONTRACT, condition_inputs,
)

log = logging.getLogger(__name__)


def reduce_sum(value, world):
    if world > 1:
        dist.all_reduce(value)
    return value


def barrier(world):
    if world > 1:
        dist.barrier()


def panel_batch(panel, idx, spec, device):
    from scripts.tau_fresh_negatives import make_batch
    batch = make_batch(panel['raw_condition'][idx], spec, device)
    batch['event_category'] = torch.as_tensor(panel['category'][idx], device=device)
    for leg in ('a', 'b'):
        for j, key in enumerate(('E', 'px', 'py', 'pz')):
            batch[f'lead_{leg}_visible_{key}'] = torch.as_tensor(panel['visible_'+leg][idx, j], device=device)
    return canonical_policy_batch(batch, contract=SUPERVISED_POLICY_CONDITIONING_CONTRACT
        if 'classification' in spec.shapes else POLICY_CONDITIONING_CONTRACT,
        num_classes=int(panel.get('num_process_classes',19)))


def cij_terms(panel, idx, delta):
    # Historical uncalibrated event-moment diagnostic. Do not change this map
    # in place: saved baselines and predeclared ablations use these exact terms.
    # The channel-binned tihsu reconstruction is logged separately below.
    from scripts.diagnose_ztautau_cij import tau_from_deltas, angles
    va, vb = panel['visible_a'][idx], panel['visible_b'][idx]
    a, b = angles(tau_from_deltas(va, delta[:, 0]), tau_from_deltas(vb, delta[:, 1]), va, vb)
    k = panel['kappas'][idx].prod(axis=1)
    if not np.isfinite(k).all() or (k == 0).any():
        raise ValueError('Missing/invalid analyzing powers in Cij panel')
    return 9*(a[:, :, None]*b[:, None, :]).reshape(-1, 9)/k[:, None]


def summarize_cij(truth, generated, weights):
    """One contribution/event, averaged over K; no classifier reweighting."""
    p, q = [np.average(x, weights=weights, axis=0).reshape(3, 3) for x in (truth, generated)]
    delta = q-p
    return dict(truth=p.tolist(), generated=q.tolist(), error=float(np.linalg.norm(delta)),
        diagonal_error=float(np.linalg.norm(np.diag(delta))),
        offdiagonal_error=float(np.linalg.norm(delta[~np.eye(3, dtype=bool)])))


def fit_head(cfg, arrays, path, device, rank, world, emit):
    """Balanced paired BCE, global fit weights, padding with ZERO weight."""
    from scripts.train_conditional_spin_ratio import build_classifier, score_pair, pair_metrics
    from scripts.tau_residual_ratio import epoch_batches, BestValidation
    fi, vi = [np.flatnonzero(arrays['split'] == j) for j in (0, 1)]
    if min(len(fi), len(vi)) < 2 or not all(arrays['event_weight'][i].sum() > 0 for i in (fi, vi)):
        raise ValueError('Empty/zero-weight classifier split')
    if not np.isfinite(arrays['event_weight']).all() or (arrays['event_weight'] < 0).any():
        raise ValueError('Paired BCE requires finite nonnegative event weights')
    w = arrays['event_weight'][fi].astype(float)
    w /= w.mean()
    torch.manual_seed(cfg['seed'])
    bare = build_classifier(cfg).to(device)
    model = torch.nn.parallel.DistributedDataParallel(bare, device_ids=[device.index]
        if device.type == 'cuda' else None) if world > 1 else bare
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    decay_epochs = int(cfg.get('lr_decay_epochs', cfg['epochs']))
    if decay_epochs < 1:raise ValueError('lr_decay_epochs must be positive')
    plateau_lr = cfg.get('scheduler') == 'validation_plateau'
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
        factor=cfg['lr_factor'], patience=cfg['lr_patience'], threshold=cfg['min_delta'],
        threshold_mode='abs', min_lr=cfg['min_lr']) if plateau_lr else
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, decay_epochs, eta_min=cfg['min_lr']))
    selector = BestValidation(cfg['patience'], cfg['min_delta'], cfg['min_steps'])
    status = torch.zeros(6, device=device, dtype=torch.float64)
    steps, best, start_epoch, resume_mode = 0, None, 0, 'fresh'
    state_path = path.with_name('fit_last.pt')
    checkpoint_fit = cfg.get('checkpoint_fit', False)
    # Bind optimizer continuation to the exact paired data, split and fit configuration.
    # A replayed policy may have different candidates even at the same policy step.
    digest = hashlib.sha256()
    if checkpoint_fit:
        for key in sorted(arrays):
            a = np.ascontiguousarray(arrays[key])
            digest.update(key.encode()); digest.update(str((a.shape, a.dtype)).encode())
            digest.update(memoryview(a).cast('B'))
    signature = digest.hexdigest()
    contract = {k: v for k, v in cfg.items() if k not in ('epochs', 'allow_best_warm_start')}
    if checkpoint_fit and state_path.exists():
        saved_fit = torch.load(state_path, map_location='cpu', weights_only=False)
        if saved_fit['contract'] != contract or saved_fit['world'] != world:
            raise ValueError('Classifier continuation configuration/world differs; explicit migration required')
        if saved_fit['data_signature'] == signature:
            bare.load_state_dict(saved_fit['model'], strict=True)
            optimizer.load_state_dict(saved_fit['optimizer'])
            scheduler.load_state_dict(saved_fit['scheduler'])
            selector.__dict__.update(saved_fit['selector'])
            steps, start_epoch, best = saved_fit['steps'], saved_fit['epoch'], saved_fit['best']
            rng = saved_fit['rng'][rank]
            torch.set_rng_state(rng['cpu'])
            if device.type == 'cuda': torch.cuda.set_rng_state(rng['cuda'], device)
            resume_mode = 'full_state'
        elif not cfg.get('allow_best_warm_start'):
            raise ValueError('Classifier continuation candidates/data differ')
        else:
            resume_mode = 'best_weights_new_candidates'
    if checkpoint_fit and resume_mode != 'full_state' and path.exists():
        if not cfg.get('allow_best_warm_start'):
            raise ValueError('Only classifier weights exist; explicitly enable best-weight warm start')
        previous = torch.load(path, map_location='cpu', weights_only=True)
        if previous.get('source_policy_step') != cfg.get('source_policy_step'):
            raise ValueError('Warm-start classifier belongs to a different policy step')
        bare.load_state_dict(previous['state_dict'], strict=True)
        start_epoch, steps = previous['epoch'], previous['optimizer_steps']
        if resume_mode == 'fresh': resume_mode = 'legacy_best_weights_new_optimizer'
        # Re-evaluate the initial weights on THIS validation split; never retain
        # a best BCE from older candidates. Dropout is disabled by score_pair.
        if rank == 0:
            p, q = score_pair(bare, arrays['condition'][vi], arrays['candidate_truth'][vi],
                arrays['candidate_generated'][vi], device, cfg['batch_size'])
            m = pair_metrics(p, q, arrays['event_weight'][vi])
            selector.update(m['bce'], steps)
            best = dict(cfg, state_dict=copy.deepcopy(previous['state_dict']), epoch=start_epoch,
                optimizer_steps=steps, val_bce=m['bce'])
        barrier(world)
    if start_epoch >= cfg['epochs'] and not (resume_mode == 'full_state' and saved_fit['stopped']):
        raise ValueError('Classifier continuation needs a larger epoch safety limit')
    if rank == 0:
        log.info('[DGPO/tau fit] mode=%s start_epoch=%s steps=%s batch_per_gpu=%s scheduler=%s',
            resume_mode, start_epoch, steps, cfg['batch_size'], cfg.get('scheduler', 'cosine'))
    already_stopped = resume_mode == 'full_state' and saved_fit['stopped']
    if already_stopped: status[3] = 1
    epoch = start_epoch-1
    for epoch in range(start_epoch, start_epoch if already_stopped else cfg['epochs']):
        model.train()
        total = torch.zeros(2, device=device, dtype=torch.float64)
        lr = optimizer.param_groups[0]['lr']
        for ii, valid, count in epoch_batches(len(fi), world, cfg['batch_size'], rank, cfg['seed'], epoch):
            idx = fi[ii]
            c, t, g = [torch.as_tensor(arrays[k][idx], device=device) for k in ('condition', 'candidate_truth', 'candidate_generated')]
            weight = torch.as_tensor(w[ii]*valid, device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            p, q = model(torch.cat((c, c)), torch.cat((t, g))).chunk(2)
            local_sum = (.5*weight*(F.softplus(-p)+F.softplus(q))).sum()
            loss = local_sum*world/count
            finite = torch.isfinite(loss).to(torch.int32)
            if world > 1:
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite.item():
                raise FloatingPointError('Nonfinite tau paired BCE')
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
            optimizer.step()
            steps += 1
            total[0] += local_sum.detach().double(); total[1] += valid.sum()
        reduce_sum(total, world)
        local = vi[rank::world]
        p, q = (score_pair(bare, arrays['condition'][local], arrays['candidate_truth'][local],
            arrays['candidate_generated'][local], device, cfg['batch_size']) if len(local) else (np.empty(0), np.empty(0)))
        gathered = [None]*world if rank == 0 else None
        if world > 1:
            dist.gather_object((local, p, q), gathered, dst=0)
        else:
            gathered = [(local, p, q)]
        if rank == 0:
            ids = np.concatenate([r[0] for r in gathered])
            m = pair_metrics(np.concatenate([r[1] for r in gathered]), np.concatenate([r[2] for r in gathered]), arrays['event_weight'][ids])
            improved, stop = selector.update(m['bce'], steps)
            eligible_improved = (steps >= cfg.get('min_selected_steps', 0)
                and (best is None or m['bce'] < best['val_bce']))
            if eligible_improved:
                best = dict(cfg, state_dict={k: v.detach().cpu().clone() for k, v in bare.state_dict().items()},
                            epoch=epoch+1, optimizer_steps=steps, val_bce=m['bce'])
                temporary = path.with_suffix('.pending.pt')
                torch.save(best, temporary); temporary.replace(path)
            selected_bce = best['val_bce'] if best is not None else m['bce']
            status[:] = torch.tensor([m['bce'], m['auc'], selected_bce, stop, selector.stale, eligible_improved], device=device)
        if world > 1:
            dist.broadcast(status, src=0)
        if rank == 0:
            emit(dict(epoch=epoch+1, optimizer_steps=steps, train_bce=float(total[0]/total[1]),
                val_bce=float(status[0]), val_auc=float(status[1]), best_val_bce=float(status[2]),
                lr=lr, grad_norm=float(grad), early_stopped=int(status[3])))
        # Do not let cosine LR rise again when fitting beyond its decay horizon.
        if plateau_lr:
            scheduler.step(float(status[0]))
        elif scheduler.last_epoch < decay_epochs:
            scheduler.step()
        if checkpoint_fit:
            rng = dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(device)
                       if device.type == 'cuda' else None)
            states = [None]*world if rank == 0 else None
            if world > 1: dist.gather_object(rng, states, dst=0)
            else: states = [rng]
            if rank == 0:
                payload = dict(version=1, contract=contract, data_signature=signature, world=world,
                    model=bare.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                    selector=dict(selector.__dict__), epoch=epoch+1, steps=steps, best=best,
                    rng=states, stopped=bool(status[3]), resume_mode=resume_mode)
                temporary = state_path.with_suffix('.pending.pt')
                torch.save(payload, temporary); temporary.replace(state_path)
        if bool(status[3]):
            break
    barrier(world)
    if checkpoint_fit:
        # fit_last is the atomic source of truth, including the selected head.
        best = torch.load(state_path, map_location='cpu', weights_only=False)['best']
        if rank == 0:
            temporary = path.with_suffix('.pending.pt')
            torch.save(best, temporary); temporary.replace(path)
    else:
        best = torch.load(path, map_location='cpu', weights_only=True)
    bare.load_state_dict(best['state_dict'], strict=True)
    bare.eval().requires_grad_(False)
    info = dict(total_steps=steps, epochs=epoch+1, best_epoch=best['epoch'], best_steps=best['optimizer_steps'],
                best_val_bce=best['val_bce'], minimum_fit_steps_met=steps >= cfg['min_steps'],
                early_stopped=bool(status[3]), resume_mode=resume_mode)
    if rank == 0:
        path.with_suffix('.json').write_text(json.dumps(info, indent=2)+'\n')
    if cfg.get('require_plateau', False) and not info['early_stopped']:
        raise RuntimeError('Classifier hit its epoch safety limit without validation plateau; best.pt preserved, reward not installed')
    return bare, best, info


class ConditionalTauCycle:
    def __init__(self, cfg, reward, actor, reference, sampler, device, rank, world, log_metrics):
        self.cfg, self.reward, self.actor, self.reference, self.sampler = cfg, reward, actor, reference, sampler
        self.device, self.rank, self.world, self.emit = device, rank, world, log_metrics
        self.fixed_round = None
        combined = bool(reward.installed_head.get('general_spin_angles'))
        if combined != bool(cfg.get('combined_reward', False)):
            raise ValueError('Combined reward requires its explicit lifecycle')
        if cfg.get('fixed_general_round', False):
            from .tau_fixed_round import FixedGeneralRound
            self.fixed_round = FixedGeneralRound(reward, reference)
        if world != int(cfg['workers']):
            raise ValueError('Conditional tau cycle requires the configured 16-GPU group')
        self.output = Path(cfg['output']); self.output.mkdir(parents=True, exist_ok=True)
        self.validation = self.read_panel(cfg['validation_panel'])
        if combined:
            from scripts.tau_general_spin_angles import validate_truth_cache
            validate_truth_cache(self.validation, reward.installed_head['general_spin_angles'])
        self.train = None
        self._training_path_verified = False
        self._verify_replay()

    @torch.no_grad()
    def verify_training_batch(self, batch, step):
        """Paired real-batch probe: incoming labels, canonical labels, panel replay.

        Read-only, once per launch. Restore RNG and actor mode. No policy update
        or classifier fit. Supervised labels must give identical live/panel
        samples on identical events/noise.
        """
        if self._training_path_verified:
            return
        from RL.DGPO_neutrino.sampling import generate_neutrino_candidates
        from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import pack_event_inputs
        from scripts.tau_fresh_negatives import isolated_rng
        n = min(32, len(batch['x']))
        raw = {k: v[:n] if torch.is_tensor(v) and v.ndim and len(v) == len(batch['x']) else v
               for k, v in batch.items()}
        fixed = canonical_policy_batch(raw, contract=self.reward.policy_conditioning_contract)
        packed, _ = pack_event_inputs(raw, self.reward.spec)
        panel = dict(raw_condition=packed.cpu().numpy(), category=raw['event_category'].cpu().numpy())
        for leg in ('a', 'b'):
            panel['visible_'+leg] = torch.stack([raw[f'lead_{leg}_visible_{key}'].reshape(-1)
                for key in ('E', 'px', 'py', 'pz')], -1).cpu().numpy()
        replay = panel_batch(panel, np.arange(n), self.reward.spec, self.device)
        draws, values, old_mode = [], [], self.actor.training
        try:
            self.actor.eval()
            for b in (raw, fixed, replay):
                with isolated_rng(self.device, 8291+self.rank), feature_precision():
                    d = generate_neutrino_candidates(self.actor, b, self.sampler, K=1,
                        num_ddim_steps=20, device=self.device, parallel_chains=1)
                    draws.append(d)
                    values.append(self.reward.compute(d, b).double().sum())
        finally:
            self.actor.train(old_mode)
        close = torch.tensor(int(torch.allclose(draws[1], draws[2], atol=1e-4, rtol=1e-4)),device=self.device)
        error = (draws[1]-draws[2]).abs().max()
        if self.world > 1:
            dist.all_reduce(close, op=dist.ReduceOp.MIN)
            dist.all_reduce(error, op=dist.ReduceOp.MAX)
        if not close.item():
            raise ValueError('Tau DGPO live training and panel sampler differ on paired events/noise')
        incoming = raw.get('classification', torch.zeros(n, device=self.device))
        sums = torch.stack([*values, incoming.ne(0).double().sum(), values[0].new_tensor(n)])
        reduce_sum(sums, self.world)
        if self.rank == 0:
            supervised = self.reward.policy_conditioning_contract == SUPERVISED_POLICY_CONDITIONING_CONTRACT
            self.emit({'tau/startup/supervised_process_labels': int(supervised),
                'tau/startup/incoming_label_nonzero_fraction': float(sums[3]/sums[4]),
                'tau/startup/paired_original_label_reward': float(sums[0]/sums[4]),
                'tau/startup/paired_canonical_reward': float(sums[1]/sums[4]),
                'tau/startup/paired_panel_reward': float(sums[2]/sums[4]),
                'tau/startup/live_panel_max_sample_error': float(error)}, step)
        self._training_path_verified = True

    @torch.no_grad()
    def verify_initial_policy(self):
        """Zero-initialized actor conditioning must preserve the ratio denominator."""
        from RL.DGPO_neutrino.sampling import generate_neutrino_candidates
        from scripts.tau_fresh_negatives import isolated_rng
        idx = np.arange(self.rank, min(16*self.world, len(self.validation['source_ids'])), self.world)
        b = panel_batch(self.validation, idx, self.reward.spec, self.device)
        samples, old_mode = [], self.actor.training
        try:
            self.actor.eval()
            for policy in (self.actor, self.reward.backbone):
                with isolated_rng(self.device, 921+self.rank), feature_precision():
                    samples.append(generate_neutrino_candidates(policy, b, self.sampler, K=1,
                        num_ddim_steps=20, device=self.device, parallel_chains=1))
        finally:
            self.actor.train(old_mode)
        if not torch.allclose(*samples, atol=1e-4, rtol=1e-4):
            raise ValueError('Initial actor no longer generates like the saved ratio denominator; do not reuse this head')
        error = (samples[0]-samples[1]).abs().max()
        if self.world > 1:
            dist.all_reduce(error, op=dist.ReduceOp.MAX)
        if self.rank == 0:
            self.emit({'tau/startup/source_policy_max_sample_error': float(error),
                       'tau/startup/baseline_head_reused': int(
                           self.reward.policy_conditioning_contract == POLICY_CONDITIONING_CONTRACT)}, 0)

    @staticmethod
    def read_panel(path):
        with np.load(path, allow_pickle=False) as f:
            return {k: f[k] for k in f.files}

    def _verify_replay(self):
        if self.reward.policy_conditioning_contract == SUPERVISED_POLICY_CONDITIONING_CONTRACT:
            for start in range(0, len(self.validation['source_ids']), 2048):
                idx = np.arange(start, min(start+2048, len(self.validation['source_ids'])))
                b = panel_batch(self.validation, idx, self.reward.spec, self.device)
                c = self.reward.prepare_inputs(b)['condition']
                np.testing.assert_allclose(c, self.validation['condition'][idx], rtol=1e-6, atol=1e-6)
            return
        with np.load(self.reward.bundle['replay'], allow_pickle=False) as replay:
            idx = np.arange(self.rank, len(replay['deltas']), self.world)
            b = panel_batch(self.validation, idx, self.reward.spec, self.device)
            delta = torch.as_tensor(replay['deltas'][idx], device=self.device)
            c, f = self.reward.features(b, delta)
            if not np.allclose(c, self.validation['condition'][idx], atol=1e-6, rtol=1e-6):
                raise ValueError('Online condition replay failed')
            if not np.allclose(f, replay['features'][idx], atol=1e-3, rtol=1e-4):
                raise ValueError('Online raw1110 candidate feature replay failed')
            _, truth_features = self.reward.features(b,
                torch.as_tensor(self.validation['truth_deltas'][idx], device=self.device))
            if not np.allclose(truth_features, self.validation['candidate_truth'][idx], atol=1e-3, rtol=1e-4):
                raise ValueError('Cached truth feature replay failed')
            # On resumed runs the installed head may differ; replay the original head.
            from scripts.train_conditional_spin_ratio import build_classifier
            from scripts.tau_fresh_negatives import isolated_rng
            with isolated_rng(self.device, 42), feature_precision():
                original = build_classifier(self.reward.bundle['head']).to(self.device).eval()
                original.load_state_dict(self.reward.bundle['head']['state_dict'], strict=True)
                with torch.no_grad():
                    score = original(torch.as_tensor(c, device=self.device), torch.as_tensor(f, device=self.device)).cpu().numpy()
            if not np.allclose(score, replay['logits'][idx], atol=1e-3, rtol=1e-4):
                raise ValueError('Saved best-val tau head logit replay failed')
        barrier(self.world)

    @torch.no_grad()
    def _prepare_truth_features(self, panel, label):
        """Cache paired truth through the SAME frozen, label-aware FiLM trunk."""
        spec = self.reward.installed_head.get('general_spin_angles')
        if spec:
            from scripts.tau_general_spin_angles import validate_truth_cache
            validate_truth_cache(panel, spec)
        if bool(panel.get('truth_features_ready', False)):
            if panel['candidate_truth'].shape[1] != self.reward.installed_head['candidate_dim']:
                raise ValueError('Cached truth features do not match reward architecture')
            return
        folder = self.output/f'initial-{label}-truth-features'
        folder.mkdir(parents=True, exist_ok=True)
        idx = np.arange(self.rank, len(panel['source_ids']), self.world)
        features = []
        for start in range(0, len(idx), self.cfg['generation_batch_size']):
            take = idx[start:start+self.cfg['generation_batch_size']]
            b = panel_batch(panel, take, self.reward.spec, self.device)
            _, f = self.reward.features(b, torch.as_tensor(panel['truth_deltas'][take], device=self.device))
            if spec:
                from scripts.tau_general_spin_angles import truth_features
                f[:, spec['base_candidate_dim']:] = truth_features(panel, take)
            features.append(f)
        np.savez(folder/f'rank-{self.rank:02d}.npz', positions=idx, features=np.concatenate(features))
        barrier(self.world)
        merged = np.empty((len(panel['source_ids']), self.reward.installed_head['candidate_dim']), np.float32)
        seen = np.zeros(len(merged), int)
        for rank in range(self.world):
            with np.load(folder/f'rank-{rank:02d}.npz', allow_pickle=False) as data:
                pos = data['positions']; np.add.at(seen, pos, 1)
                merged[pos] = data['features']
        if not np.all(seen == 1) or not np.isfinite(merged).all():
            raise ValueError('Incomplete/nonfinite supervised truth feature cache')
        panel.update(candidate_truth=merged, truth_features_ready=np.array(True))
        if self.rank == 0:
            path = Path(self.cfg[label+'_panel'])
            temporary = path.with_suffix('.pending.npz')
            np.savez(temporary, **panel); temporary.replace(path)
        barrier(self.world)

    def bootstrap_supervised_reward(self, epoch, step):
        """Fresh paired classifier fit before any actor update; resume keeps it."""
        if self.reward.policy_conditioning_contract != SUPERVISED_POLICY_CONDITIONING_CONTRACT:
            return False
        if self.cfg.get('policy_conditioning_contract') != self.reward.policy_conditioning_contract:
            raise ValueError('Actor/classifier process-label contracts differ')
        if self.cfg.get('combined_reward', False):
            from .tau_combined_reward import verify_saved_scores
            if (not bool(self.validation.get('truth_features_ready', False))
                    or self.validation['candidate_truth'].shape[1] != self.reward.installed_head['candidate_dim']):
                raise ValueError('Combined validation truth feature cache is incomplete')
            verify_saved_scores(self, step)
            conditioning = getattr(getattr(self.actor,'TruthGeneration',None),'visible_conditioning',None)
            if step == 0 and getattr(conditioning,'met_conditioning',None) is not None:
                self.verify_initial_policy()
            return False
        self.train = self.read_panel(self.cfg['train_panel'])
        self._prepare_truth_features(self.train, 'train')
        self._prepare_truth_features(self.validation, 'validation')
        if self.reward.installed_head.get('fitting_current_policy', False):
            return False
        if step != 0 or self.reward.round_id != 0:
            raise ValueError('Unfitted initial classifier cannot resume after actor updates')
        self.verify_initial_policy()
        self.refit(epoch, step, force=True)
        if self.reward.installed_head['optimizer_steps'] < self.cfg['fit']['min_steps']:
            raise ValueError('Initial process-conditioned classifier is undertrained')
        return True

    @torch.no_grad()
    def generate(self, panel, output, candidates, seed, physics, *, retain_features=True):
        # Classifier fitting/audits keep the historical feature bundle. The
        # read-only direction diagnostic only consumes all-K rewards and raw
        # physics arrays; avoid a redundant first-candidate feature pass and
        # sixteen workers rereading that large unused bundle on every probe.
        if type(retain_features) is not bool or (not retain_features and not physics):
            raise ValueError('Feature-free generation requires a physics/reward evaluation')
        from RL.DGPO_neutrino.sampling import generate_neutrino_candidates
        from scripts.tau_fresh_negatives import isolated_rng
        output.mkdir(parents=True, exist_ok=True)
        idx = np.arange(self.rank, len(panel['source_ids']), self.world)
        feature, draws, rewards, terms = [], [], [], []
        old_mode = self.actor.training
        try:
            with isolated_rng(self.device, seed+self.rank):
                self.actor.eval()
                for start in range(0, len(idx), self.cfg['generation_batch_size']):
                    ii = idx[start:start+self.cfg['generation_batch_size']]
                    b = panel_batch(panel, ii, self.reward.spec, self.device)
                    delta = generate_neutrino_candidates(self.actor, b, self.sampler, K=candidates,
                        num_ddim_steps=20, device=self.device, parallel_chains=1)
                    if retain_features:
                        _, f = self.reward.features(b, delta[0])
                        feature.append(f)
                    draws.append(delta.permute(1, 0, 2, 3).cpu().numpy())
                    if physics:
                        rewards.append(self.reward.compute(delta, b).T.cpu().numpy())
                        if not self.cfg.get('direct_oracle_monitor', False):
                            terms.append(np.mean([cij_terms(panel, ii, d.cpu().numpy()) for d in delta], axis=0))
                    log.info('[DGPO/tau generation] %s rank=%s events=%s/%s K=%s', output.name, self.rank, start+len(ii), len(idx), candidates)
        finally:
            self.actor.train(old_mode)
        data = dict(positions=idx, deltas=np.concatenate(draws))
        if retain_features:
            data['features'] = np.concatenate(feature)
        if physics:
            data.update(rewards=np.concatenate(rewards))
            if terms:
                data['cij'] = np.concatenate(terms)
        np.savez(output/f'rank-{self.rank:02d}.npz', **data)
        barrier(self.world)
        # All ranks need head inputs; only rank0 collects K-sample physics arrays.
        merged, seen = {}, np.zeros(len(panel['source_ids']), int)
        physics_keys = ['rewards', 'deltas'] + ([] if self.cfg.get('direct_oracle_monitor', False) else ['cij'])
        keys = (['features'] if retain_features else []) + (physics_keys if physics and self.rank == 0 else [])
        for rank in range(self.world):
            with np.load(output/f'rank-{rank:02d}.npz', allow_pickle=False) as f:
                pos = f['positions']; np.add.at(seen, pos, 1)
                for key in keys:
                    if key not in merged:
                        merged[key] = np.empty((len(seen), *f[key].shape[1:]), dtype=f[key].dtype)
                    merged[key][pos] = f[key]
        if not np.all(seen == 1) or any(not np.isfinite(v).all() for v in merged.values()):
            raise ValueError('Incomplete/nonfinite distributed candidate panel')
        return merged

    def fit(self, panel, generated, folder, *, audit, step, head_overrides=None, log_phase=None,
            reward_ratio_bound=30, min_selected_steps=0, log_step=None):
        from scripts.tau_residual_ratio import split_events
        from scripts.tau_fresh_negatives import isolated_rng
        if isinstance(reward_ratio_bound, bool) or reward_ratio_bound not in (30, None):
            raise ValueError('Declare reward_ratio_bound as 30 or None')
        data = {k: panel[k] for k in ('source_ids', 'condition', 'candidate_truth', 'event_weight', 'split')}
        data['candidate_generated'] = generated['features']
        if audit:
            data['split'] = split_events(data['source_ids'], self.cfg['audit_seed'], (.6, .2, .2))
        cfg = copy.deepcopy(self.reward.bundle['head'])
        cfg.pop('state_dict')
        if self.cfg.get('combined_reward', False) and audit:
            from .tau_combined_reward import general_audit_inputs
            cfg, data = general_audit_inputs(cfg, data)
        cfg.update(self.cfg['fit'])
        if self.cfg.get('production_cross_attention', False):
            cfg.update(cross_attention=True, attention_heads=4, head_depth=3)
        if head_overrides:
            if not self.cfg.get('classifier_only') or set(head_overrides) - {'head_depth', 'cross_attention', 'attention_heads'}:
                raise ValueError('Architecture overrides restricted to isolated classifier depth experiment')
            cfg.update(head_overrides)
        cfg.update(seed=self.cfg['audit_seed'] if audit else self.cfg['refit_seed'], ratio_objective='bce',
                   ratio_bound=None if audit else reward_ratio_bound, source_policy_step=step, fitting_current_policy=True)
        if type(min_selected_steps) is not int or min_selected_steps < 0:
            raise ValueError('min_selected_steps must be a nonnegative integer')
        cfg['min_selected_steps'] = max(min_selected_steps, int(self.cfg.get('minimum_selected_steps', 0)))
        folder.mkdir(parents=True, exist_ok=True)
        phase = log_phase or ('audit_fit' if audit else 'refit_fit')
        def emit(row):
            self.emit({**{f'tau/{phase}/{k}': v for k, v in row.items()},
                       f'tau/{phase}/log_index': step*cfg['epochs']+row['epoch'],
                       f'tau/{phase}/policy_step': step, f'tau/{phase}/reward_round': self.reward.round_id},
                      step if log_step is None else log_step)
        with isolated_rng(self.device, cfg['seed']), feature_precision():
            model, saved, status = fit_head(cfg, data, folder/'best.pt', self.device, self.rank, self.world, emit)
        return model, saved, status, data

    def classifier_only(self, step):
        """Fresh best-val head on current-policy negatives; never updates actor/reference."""
        self.train = self.read_panel(self.cfg['train_panel'])
        if len(self.train['source_ids']) != 416701 or len(self.validation['source_ids']) != 119002:
            raise ValueError('Expected complete filtered 10% train and independent validation panels')
        if np.intersect1d(self.train['source_ids'], self.validation['source_ids']).size:
            raise ValueError('Classifier train/validation identities overlap')
        folder = self.output/'classifier-only'
        val = self.generate(self.validation, folder/'validation_candidates', self.cfg['validation_candidates'],
                            self.cfg['validation_seed'], True)
        self.reward_cij_probe(step, val, phase='inherited')
        train = self.generate(self.train, folder/'training_candidates', 1, self.cfg['refit_seed']+step, False)
        depths = self.cfg.get('classifier_depths', [3])
        if not depths or len(set(depths)) != len(depths) or any(type(d) is not int or d < 1 for d in depths):
            raise ValueError('Classifier depths must be unique positive integers')
        original_head = self.reward.head
        try:
            if self.cfg.get('classifier_attention_ablation'):
                if step != 1780 or depths != [3]:
                    raise ValueError('Attention ablation requires step1780 and depth3')
                for attention in (False, True):
                    self._classifier_depth_arm(step, 3, folder, train, val, attention=attention)
            else:
                for depth in depths:
                    self._classifier_depth_arm(step, depth, folder, train, val)
        finally:
            self.reward.head = original_head
        del train, val
        barrier(self.world)

    def _classifier_depth_arm(self, step, depth, shared_folder, train, val, attention=False):
        arm = f'fresh_depth{depth}' + ('_attention' if attention else '')
        folder = shared_folder/arm
        model, best, status, arrays = self.fit(self.train, train, folder/'fresh_head', audit=False, step=step,
            head_overrides={'head_depth': depth, 'cross_attention': attention, 'attention_heads': 4},
            log_phase=f'classifier_only/{arm}/fit')
        # Diagnostic-only model swap; production's strict depth3 install contract
        # remains unchanged. No reference recentering or reward-round increment.
        self.reward.head = model
        from scripts.train_conditional_spin_ratio import score_pair, pair_metrics
        ii = np.arange(self.rank, len(self.validation['source_ids']), self.world)
        with feature_precision():
            p, q = score_pair(model, self.validation['condition'][ii], self.validation['candidate_truth'][ii],
                              val['features'][ii], self.device, self.cfg['fit']['batch_size'])
        with np.load(shared_folder/'validation_candidates'/f'rank-{self.rank:02d}.npz', allow_pickle=False) as saved:
            positions, deltas = saved['positions'], saved['deltas']
        if not np.array_equal(positions, ii): raise ValueError('Validation shard identities changed')
        scores = []
        with torch.no_grad():
            for start in range(0, len(ii), self.cfg['generation_batch_size']):
                take = ii[start:start+self.cfg['generation_batch_size']]
                b = panel_batch(self.validation, take, self.reward.spec, self.device)
                delta = torch.as_tensor(deltas[start:start+len(take)], device=self.device).permute(1, 0, 2, 3)
                scores.append(self.reward.compute(delta, b).T.cpu().numpy())
        np.savez(folder/f'fresh-scores-{self.rank:02d}.npz', positions=ii,
                 rewards=np.concatenate(scores), truth_logits=p, generated_logits=q)
        barrier(self.world)
        if self.rank == 0:
            all_p = np.empty(len(self.validation['source_ids'])); all_q = np.empty_like(all_p)
            for rank in range(self.world):
                with np.load(folder/f'fresh-scores-{rank:02d}.npz', allow_pickle=False) as saved:
                    pos = saved['positions']; val['rewards'][pos] = saved['rewards']
                    all_p[pos] = saved['truth_logits']; all_q[pos] = saved['generated_logits']
            metrics = pair_metrics(all_p, all_q, self.validation['event_weight'])
            report = dict(complete=True, policy_step=step, actor_updates=0, architecture_changed=depth != 3 or attention,
                          cross_attention=attention, attention_heads=4 if attention else 0,
                          attention_tokens='existing normalized visible x with mask and slot identity; not PET tokens' if attention else None,
                          head_depth=depth, condition_width=best['condition_width'],
                          parameters=sum(p.numel() for p in model.parameters()),
                          training_population=len(self.train['source_ids']),
                          optimizer_fit_events=int(np.sum(arrays['split']==0)),
                          selection_events=int(np.sum(arrays['split']==1)),
                          external_events=len(all_p), fit=status, external=metrics,
                          selection='best internal-validation BCE; external Cij not used for selection',
                          feature_backbone='unchanged frozen raw1110; generator is pinned current DGPO')
            (folder/'classifier_report.json').write_text(json.dumps(report, indent=2)+'\n')
            self.emit({**{f'tau/classifier_only/{arm}/external/{k}': v for k,v in metrics.items()},
                       **{f'tau/classifier_only/{arm}/fit/{k}': v for k,v in status.items()},
                       'tau/classifier_only/actor_updates': 0}, step)
        self.reward_cij_probe(step, val, phase=arm)
        del model
        barrier(self.world)

    def reward_cij_probe(self, step, generated=None, phase='resume'):
        """Installed reward only; no classifier fitting, refit, or clock changes."""
        from scripts.tau_reward_cij_probe import compare_reweighting
        folder = self.output/f'reward-probe-{phase}-step-{step:08d}'
        folder.mkdir(parents=True, exist_ok=True)
        g = generated if generated is not None else self.generate(
            self.validation, folder/'candidates', self.cfg['validation_candidates'],
            self.cfg['validation_seed'], True)
        if self.rank == 0:
            idx = np.arange(len(self.validation['source_ids']))
            truth = cij_terms(self.validation, idx, self.validation['truth_deltas'])
            terms = np.stack([cij_terms(self.validation, idx, g['deltas'][:, k])
                              for k in range(g['rewards'].shape[1])], axis=1)
            report = compare_reweighting(truth, terms, g['rewards'], self.validation['event_weight'])
            report.update(policy_step=step, reward_round=self.reward.round_id, phase=phase,
                scope='Fixed candidates; native bounded log-ratio. Global ratio may change condition mass; '
                'within-condition normalization preserves it but is finite-K self-normalized. '
                'Point estimates only; neither is a DGPO update or proof of capacity limitation.')
            (folder/'report.json').write_text(json.dumps(report, indent=2)+'\n')
            np.savez_compressed(folder/'measurements.npz', source_ids=self.validation['source_ids'],
                truth_cij=truth, candidate_cij=terms, log_ratio=g['rewards'], weight=self.validation['event_weight'])
            out = {'tau/reweight/policy_step': step, 'tau/reweight/reward_round': self.reward.round_id,
                   'tau/reweight/within_condition_ess_fraction': report['within_condition_ess_fraction']}
            for name, arm in report['arms'].items():
                for key, value in arm.items():
                    if key != 'cij': out[f'tau/reweight/{name}/{key}'] = value
                for i, a in enumerate('krn'):
                    for j, b in enumerate('krn'):
                        out[f'tau/reweight/{name}/cij/{a}{b}'] = arm['cij'][i][j]
                for key in ('error', 'diagonal_error', 'offdiagonal_error', 'nn_error'):
                    out[f'tau/reweight/{name}/change/{key}'] = arm[key]-report['arms']['unweighted'][key]
            self.emit(out, step)
            if phase == 'inherited' or phase.startswith(('fresh', 'transfer_')):
                self.emit({key.replace('tau/reweight/', f'tau/reweight_{phase}/'): value
                           for key, value in out.items()}, step)
            log.info('[DGPO/tau reward probe] step=%s arms=%s', step, report['arms'])
        barrier(self.world)

    def _score_transfer_head(self, generated, candidate_folder, output, step, phase):
        """Score a saved K panel, never regenerate it when changing the head."""
        from scripts.train_conditional_spin_ratio import score_pair, pair_metrics
        output.mkdir(parents=True, exist_ok=True)
        ii = np.arange(self.rank, len(self.validation['source_ids']), self.world)
        with feature_precision():
            p, q = score_pair(self.reward.head, self.validation['condition'][ii],
                self.validation['candidate_truth'][ii], generated['features'][ii],
                self.device, self.cfg['fit']['batch_size'])
        with np.load(candidate_folder/f'rank-{self.rank:02d}.npz', allow_pickle=False) as saved:
            positions, deltas = saved['positions'], saved['deltas']
        if not np.array_equal(positions, ii):
            raise ValueError('Transfer comparison changed validation identities')
        scores = []
        with torch.no_grad(), feature_precision():
            for start in range(0, len(ii), self.cfg['generation_batch_size']):
                take = ii[start:start+self.cfg['generation_batch_size']]
                batch = panel_batch(self.validation, take, self.reward.spec, self.device)
                delta = torch.as_tensor(deltas[start:start+len(take)], device=self.device).permute(1, 0, 2, 3)
                scores.append(self.reward.compute(delta, batch).T.cpu().numpy())
        np.savez(output/f'rank-{self.rank:02d}.npz', positions=ii,
                 rewards=np.concatenate(scores), truth_logits=p, generated_logits=q)
        barrier(self.world)
        metrics = None
        if self.rank == 0:
            all_p = np.empty(len(self.validation['source_ids']))
            all_q = np.empty_like(all_p)
            seen = np.zeros(len(all_p), int)
            for rank in range(self.world):
                with np.load(output/f'rank-{rank:02d}.npz', allow_pickle=False) as saved:
                    pos = saved['positions']; np.add.at(seen, pos, 1)
                    generated['rewards'][pos] = saved['rewards']
                    all_p[pos], all_q[pos] = saved['truth_logits'], saved['generated_logits']
            if not np.all(seen == 1):
                raise ValueError('Transfer comparison has missing/repeated validation identities')
            metrics = pair_metrics(all_p, all_q, self.validation['event_weight'])
            w, r = self.validation['event_weight'], generated['rewards']
            metrics.update(mean=float(np.average(r.mean(1), weights=w)),
                           within_condition_std=float(np.average(r.std(1), weights=w)))
            self.emit({f'tau/transfer/head_comparison/{phase}/{k}': v
                       for k, v in metrics.items()}, step)
        barrier(self.world)
        return metrics

    def prepare_reward_transfer(self, epoch, step):
        """One matched refit transaction before a frozen, native-update probe.

        Actor parameters and AdamW are never changed here. The inherited and
        fresh heads see identical held-out candidates. Only after comparison is
        the fresh best-val head installed and the velocity reference recentered,
        exactly as in production refitting. This is NOT a head-only causal arm.
        """
        probe = self.cfg.get('transfer_probe', {})
        if not probe.get('enabled') or step != probe['source_step']:
            raise ValueError('Tau transfer startup requires its pinned source step')
        if self.reward.round_id != probe['expected_source_reward_round']:
            raise ValueError('Pinned source classifier round changed')
        if not self.reward.installed_head.get('cross_attention', False):
            raise ValueError('Transfer source must already contain the attention classifier')
        self.train = self.read_panel(self.cfg['train_panel'])
        if len(self.train['source_ids']) != 416701 or len(self.validation['source_ids']) != 119002:
            raise ValueError('Transfer requires the complete filtered train and validation populations')
        if np.intersect1d(self.train['source_ids'], self.validation['source_ids']).size:
            raise ValueError('Transfer fitting and validation identities overlap')
        folder = self.output/'transfer-startup'
        candidate_folder = folder/'validation_candidates'
        val = self.generate(self.validation, candidate_folder, self.cfg['validation_candidates'],
                            self.cfg['validation_seed'], True)
        inherited_round, inherited_denominator = self.reward.round_id, self.reward.denominator_step
        inherited = self._score_transfer_head(val, candidate_folder, folder/'inherited_scores', step, 'inherited')
        self.reward_cij_probe(step, val, phase='transfer_inherited')
        train = self.generate(self.train, folder/'training_candidates', 1,
                              self.cfg['refit_seed']+step, False)
        model, best, status, _ = self.fit(self.train, train, folder/'fresh_head',
            audit=False, step=step, log_phase='transfer/refit_fit')
        old_head = self.reward.head
        try:
            self.reward.head = model
            fresh = self._score_transfer_head(val, candidate_folder, folder/'fresh_scores', step, 'fresh')
            self.reward_cij_probe(step, val, phase='transfer_fresh')
        finally:
            self.reward.head = old_head
        self.reward.install(best, policy_step=step, epoch=epoch)
        self.reference.load_state_dict(self.actor.state_dict(), strict=True)
        self.reference.eval().requires_grad_(False)
        if self.rank == 0:
            report = dict(complete=True, source_step=step, actor_updates=0,
                inherited_round=inherited_round, inherited_denominator_step=inherited_denominator,
                installed_round=self.reward.round_id, installed_denominator_step=self.reward.denominator_step,
                events=len(self.validation['source_ids']), candidates=self.cfg['validation_candidates'],
                inherited_external=inherited, fresh_external=fresh, fit=status,
                selection='Absolute best internal-validation BCE; external physics never selects the head',
                reference_recentered_once=True, actor_optimizer_reset=False,
                scope='Matched old/new heads on identical candidates; then normal joint head/reference refit transaction. '
                      'External point estimates, not an isolated causal attribution or certified density ratio.')
            (folder/'report.json').write_text(json.dumps(report, indent=2)+'\n')
            self.emit({'tau/transfer/startup/source_step': step,
                'tau/transfer/startup/inherited_round': inherited_round,
                'tau/transfer/startup/inherited_denominator_step': inherited_denominator,
                'tau/transfer/startup/installed_round': self.reward.round_id,
                'tau/transfer/startup/reference_recentered': 1,
                'tau/transfer/startup/actor_optimizer_reset': 0,
                **{f'tau/transfer/startup/fit/{k}': v for k, v in status.items()}}, step)
        del model, train, val
        barrier(self.world)

    def transfer_audit(self, generated, folder, step):
        """Independent best-val audit on already generated probe samples.

        Does not install a reward or advance the normal evaluation/refit clocks.
        Uses the existing unbounded paired-BCE audit and disjoint event split.
        """
        from scripts.train_conditional_spin_ratio import score_pair, pair_metrics
        model, _, status, arrays = self.fit(self.validation, generated, folder,
            audit=True, step=step, log_phase='transfer/audit_fit')
        if not status['minimum_fit_steps_met']:
            raise ValueError('Transfer audit did not reach the configured minimum fit steps')
        local = np.flatnonzero(arrays['split'] == 2)[self.rank::self.world]
        with feature_precision():
            p, q = score_pair(model, arrays['condition'][local], arrays['candidate_truth'][local],
                             arrays['candidate_generated'][local], self.device, self.cfg['fit']['batch_size'])
        gathered = [None]*self.world if self.rank == 0 else None
        if self.world > 1:
            dist.gather_object((local, p, q), gathered, dst=0)
        else:
            gathered = [(local, p, q)]
        result = None
        if self.rank == 0:
            indices = np.concatenate([entry[0] for entry in gathered])
            if sorted(indices.tolist()) != np.flatnonzero(arrays['split'] == 2).tolist():
                raise ValueError('Transfer audit test identities missing/repeated')
            result = pair_metrics(np.concatenate([entry[1] for entry in gathered]),
                np.concatenate([entry[2] for entry in gathered]), arrays['event_weight'][indices])
            result.update({f'fit_{k}': v for k, v in status.items()})
            result['test_events'] = len(indices)
            (folder/'external.json').write_text(json.dumps(result, indent=2)+'\n')
        del model
        barrier(self.world)
        return result

    def evaluate(self, epoch, step):
        if epoch <= self.reward.last_evaluation_epoch:
            return
        evaluation_started = time.perf_counter()
        folder = self.output/f'eval-step-{step:08d}'
        if self.cfg.get('direct_oracle_monitor', False):
            g = self.generate(self.validation, folder/'candidates', self.cfg['validation_candidates'],
                              self.cfg['validation_seed'], True, retain_features=False)
            if self.rank == 0:
                from scripts.tau_oracle_dgpo_monitor import export_monitor
                metrics = export_monitor(self.validation, g['deltas'], folder, self.output, step)
                if self.cfg.get('joint_oracle_monitor'):
                    from scripts.tau_joint_oracle_monitor import export_joint_monitor
                    metrics.update(export_joint_monitor(self.validation, g['deltas'], folder, self.output, step,
                                                        self.cfg['joint_oracle_monitor']))
                self.emit(metrics, step)
                # The logger accepts W&B media as well as numeric metrics.
                import wandb
                self.emit({'tau/direct/distributions': wandb.Image(str(folder/'direct_distributions.png'))}, step)
                if self.cfg.get('joint_oracle_monitor'):
                    self.emit({'tau/joint/distributions': wandb.Image(str(folder/'joint_distributions.png'))}, step)
            self.reward.last_evaluation_epoch = epoch
            barrier(self.world)
            return
        g = self.generate(self.validation, folder/'candidates', self.cfg['validation_candidates'], self.cfg['validation_seed'], True)
        if self.cfg.get('reward_cij_probe', False):
            self.reward_cij_probe(step, g, phase='validation')
        model, _, status, a = self.fit(self.validation, g, folder/'fresh_audit', audit=True, step=step)
        from scripts.train_conditional_spin_ratio import score_pair, pair_metrics
        ti = np.flatnonzero(a['split'] == 2)[self.rank::self.world]
        with feature_precision():
            p, q = score_pair(model, a['condition'][ti], a['candidate_truth'][ti], a['candidate_generated'][ti], self.device, self.cfg['fit']['batch_size'])
        np.savez(folder/f'audit-rank-{self.rank:02d}.npz', positions=ti, truth_logits=p, generated_logits=q)
        barrier(self.world)
        if self.rank == 0:
            audit = [np.load(folder/f'audit-rank-{r:02d}.npz', allow_pickle=False) for r in range(self.world)]
            try:
                indices = np.concatenate([f['positions'] for f in audit])
                if sorted(indices.tolist()) != np.flatnonzero(a['split'] == 2).tolist():
                    raise ValueError('Audit test identities missing/repeated')
                m = pair_metrics(np.concatenate([f['truth_logits'] for f in audit]),
                                 np.concatenate([f['generated_logits'] for f in audit]), a['event_weight'][indices])
            finally:
                for f in audit:
                    f.close()
            truth = cij_terms(self.validation, np.arange(len(a['source_ids'])), self.validation['truth_deltas'])
            report = summarize_cij(truth, g['cij'], a['event_weight'])
            from scripts.tau_tihsu_cij import panel_report, monitoring_metrics
            if self.cfg.get('combined_reward', False):
                from .tau_combined_reward import native_cij_monitor
                baseline = g['deltas']
                if step:
                    with np.load(self.output/'eval-step-00000000'/'measurements.npz', allow_pickle=False) as saved:
                        np.testing.assert_array_equal(saved['source_ids'], self.validation['source_ids'])
                        baseline = saved['deltas']
                tihsu_report = native_cij_monitor(self.validation, g['deltas'], baseline,
                    bootstrap=self.cfg['cij_bootstrap'], seed=42)
            else:
                tihsu_report = panel_report(self.validation, g['deltas'])
            (folder/'tihsu_reconstruction_cij.json').write_text(
                json.dumps(tihsu_report, indent=2, allow_nan=False)+'\n')
            report['legacy_cij_protocol'] = 'uncalibrated-event-moments-krn'
            report['tihsu_reconstruction'] = tihsu_report
            report.update(policy_step=step, epoch=epoch, reward_round=self.reward.round_id,
                events=len(truth), candidates=self.cfg['validation_candidates'],
                audit=m, audit_fit=status, audit_test_events=len(indices),
                scope='Native unweighted current-policy samples, same complete filtered validation; fixed-energy tau reconstruction, no calibration.',
                uncertainty='Monitoring point estimates; repeated external panel, not a pristine final test; K1 fresh audit, K>1 Cij.')
            (folder/'report.json').write_text(json.dumps(report, indent=2)+'\n')
            np.savez(folder/'measurements.npz', source_ids=a['source_ids'], truth_cij=truth, generated_cij=g['cij'],
                     weight=a['event_weight'], deltas=g['deltas'], reward=g['rewards'])
            # Retain legacy moments in local reports for historical replay, but
            # publish only the explicitly versioned reconstruction Cij to W&B.
            if self.cfg.get('wandb_profile') == 'tau_spin':
                import wandb
                from .tau_spin_logging import compact_cij_metrics
                out = compact_cij_metrics(tihsu_report, self.validation, wandb)
            else:
                out = monitoring_metrics(tihsu_report, 'tau/tihsu_reconstruction/cij')
            if 'native_change' in tihsu_report:
                change = tihsu_report['native_change']
                out.update({f'tau/tihsu_reconstruction/cij/{k}': change[k] for k in ('baseline', 'current', 'delta')})
                out.update({'tau/tihsu_reconstruction/cij/change_ci_low': change['ci95'][0],
                            'tau/tihsu_reconstruction/cij/change_ci_high': change['ci95'][1],
                            'tau/fresh_audit/general_only': 1})
            r, w = g['rewards'], a['event_weight']
            soft = np.exp(r-r.max(axis=1, keepdims=True)); soft /= soft.sum(axis=1, keepdims=True)
            out.update({f'tau/fresh_audit/{k}': v for k, v in m.items()})
            out.update({f'tau/fresh_audit/{k}': v for k, v in status.items()})
            out.update({'tau/policy_step': step, 'tau/reward_round': self.reward.round_id,
                'tau/validation/events': len(truth), 'tau/validation/candidates': r.shape[1],
                'tau/reward/mean': float(np.average(r.mean(1), weights=w)),
                'tau/reward/best_of_k': float(np.average(r.max(1), weights=w)),
                'tau/reward/worst_of_k': float(np.average(r.min(1), weights=w)),
                'tau/reward/within_condition_std': float(np.average(r.std(1), weights=w)),
                'tau/reward/within_condition_ess_fraction': float(np.average(1/(r.shape[1]*(soft**2).sum(1)), weights=w))})
            from RL.DGPO_neutrino.diagnostics.tau_marginals import monitor
            marginal_metrics, plots = monitor(self.validation, g['deltas'],
                self.output/'eval-step-00000000'/'measurements.npz', folder)
            out.update(marginal_metrics)
            if not plots:
                log.warning('Step-zero samples missing: 1D baseline monitoring unavailable; not substituting refit reference')
            else:
                import wandb
                out.update({f'tau/marginal/plots/{name}': wandb.Image(str(path)) for name, path in plots.items()})
            if self.cfg.get('wandb_profile') == 'tau_spin':
                out['tau/fresh_audit/auc_gap'] = abs(m['auc']-.5)
                cap = self.reward.installed_head['ratio_bound']
                if cap is not None:
                    out['tau/reward/near_cap_fraction'] = float(np.average(
                        (np.exp(r) >= .9*cap).mean(1), weights=w))
                out['tau/validation/seconds'] = time.perf_counter() - evaluation_started
            self.emit(out, step)
        del model, g
        self.reward.last_evaluation_epoch = epoch
        barrier(self.world)

    def initialize_adaptive_oracle(self, step):
        cfg = self.cfg.get('adaptive_oracle')
        state = getattr(self.reward, 'adaptive_oracle_state', None)
        if cfg is None:
            if state is not None:
                raise ValueError('Adaptive checkpoint cannot resume into fixed oracle experiment')
            return
        from .tau_adaptive_oracle import live_alpha
        if state is not None:
            live_alpha(dict(alpha=cfg['max_alpha'], adaptive=cfg), self.reward)
            return
        if step != 0 or self.reward.round_id != 0:
            raise ValueError('Cannot recalibrate missing adaptive state after policy updates')
        self.reward.adaptive_oracle_state = self.calibrate_adaptive_oracle(
            self.reward.head, step, self.reward.round_id)

    def calibrate_adaptive_oracle(self, model, step, round_id):
        """Independent fixed external-validation subset; no fit or head selection.

        Generates only K1 for 11,903 calibration events once per accepted refit.
        Full raw-monitor plots remain exploratory; export reserved IDs so final
        physics/unfolding comparisons can exclude these feedback events.
        """
        cfg = self.cfg.get('adaptive_oracle')
        if cfg is None:
            return None
        from .tau_adaptive_oracle import validate_gate, paired_auc, next_state
        from scripts.train_conditional_spin_ratio import score_pair
        validate_gate(cfg)
        count = len(self.validation['source_ids'])
        selected = np.sort(np.random.default_rng(cfg['calibration_seed']).permutation(count)[:cfg['calibration_events']])
        if len(selected) != cfg['calibration_events']:
            raise ValueError('Incomplete adaptive calibration panel')
        panel = {k: (v[selected] if isinstance(v, np.ndarray) and v.ndim and len(v)==count else v)
                 for k,v in self.validation.items()}
        folder = self.output/f'oracle-calibration-step-{step:08d}'
        generated = self.generate(panel, folder/'candidates', 1, cfg['generation_seed'], False)
        idx = np.arange(self.rank, len(selected), self.world)
        with feature_precision():
            p,q = score_pair(model, panel['condition'][idx], panel['candidate_truth'][idx],
                            generated['features'][idx], self.device, self.cfg['fit']['batch_size'])
        gathered = [None]*self.world if self.rank==0 else None
        if self.world>1: dist.gather_object((idx,p,q),gathered,dst=0)
        else: gathered=[(idx,p,q)]
        payload=[None]
        if self.rank==0:
            # Broadcast failures too: never strand other workers at a collective.
            try:
                positions=np.concatenate([v[0] for v in gathered])
                if sorted(positions.tolist()) != list(range(len(selected))):
                    raise ValueError('Missing/repeated gate calibration events')
                p=np.concatenate([v[1] for v in gathered]);q=np.concatenate([v[2] for v in gathered])
                state=next_state(cfg,getattr(self.reward,'adaptive_oracle_state',None),paired_auc(p,q),round_id=round_id,step=step)
                np.savez(folder/'scores.npz',source_ids=panel['source_ids'][positions],truth_logits=p,generated_logits=q)
                np.savez(self.output/'oracle_calibration_ids.npz',source_ids=panel['source_ids'],positions=selected)
                temporary=folder/'gate.pending.json';temporary.write_text(json.dumps(state,indent=2));temporary.replace(folder/'gate.json')
                self.emit({**{'oracle/adaptive/'+k:state[k] for k in ('auc','auc_se','resolved_gap','baseline_gap','alpha','round_id','policy_step','events')},
                           'oracle/adaptive/refit_calibration':int(round_id>0)},step)
                payload[0]=dict(state=state)
            except Exception as exc:
                payload[0]=dict(error=str(exc))
        if self.world>1: dist.broadcast_object_list(payload,src=0)
        if 'error' in payload[0]:raise ValueError(payload[0]['error'])
        return payload[0]['state']

    def refit(self, epoch, step, *, force=False):
        if self.cfg.get('fixed_general_round', False):
            raise ValueError('Fixed General round forbids classifier refit and reference recentering')
        if not force and epoch <= self.reward.last_refit_epoch:
            return
        refit_started = time.perf_counter()
        if self.train is None:
            self.train = self.read_panel(self.cfg['train_panel'])
            if self.cfg.get('combined_reward', False):
                from scripts.tau_general_spin_angles import validate_truth_cache
                validate_truth_cache(self.train, self.reward.installed_head['general_spin_angles'])
        if len(self.train['source_ids']) != 416701 or np.intersect1d(
                self.train['source_ids'], self.validation['source_ids']).size:
            raise ValueError('Refit requires filtered 10% training population disjoint from external validation')
        folder = self.output/f'refit-step-{step:08d}'
        g = self.generate(self.train, folder/'candidates', 1, self.cfg['refit_seed']+step, False)
        model, best, status, _ = self.fit(self.train, g, folder, audit=False, step=step)
        if self.reward.policy_conditioning_contract == SUPERVISED_POLICY_CONDITIONING_CONTRACT:
            if (best['optimizer_steps'] < self.cfg['minimum_selected_steps']
                    or not status['minimum_fit_steps_met']):
                raise ValueError('Do not install an undertrained process-conditioned classifier')
        # Complete best-head + denominator transaction on EVERY rank. No actor
        # update happens here; no mixing old ratios or resetting actor AdamW.
        adaptive_state = self.calibrate_adaptive_oracle(model, step, self.reward.round_id+1)
        self.reward.install(best, policy_step=step, epoch=epoch)
        if adaptive_state is not None:
            self.reward.adaptive_oracle_state = adaptive_state
        self.reference.load_state_dict(self.actor.state_dict(), strict=True)
        self.reference.eval().requires_grad_(False)
        if self.rank == 0:
            self.emit({**{f'tau/refit/{k}': v for k, v in status.items()},
                'tau/refit/policy_step': step, 'tau/refit/reward_round': self.reward.round_id,
                'tau/refit/reference_recentered': 1, 'tau/refit/actor_optimizer_reset': 0}, step)
        if self.rank == 0 and self.cfg.get('wandb_profile') == 'tau_spin':
            self.emit({'tau/refit/seconds': time.perf_counter()-refit_started}, step)
        del model, g
        barrier(self.world)

    def epoch_end(self, epoch, step, *, final=False):
        if (self.cfg.get('mechanism_probe') or {}).get('enabled', False):
            # Fixed teacher/reference interventions have their own observer.
            return False
        if self.cfg.get('transfer_probe', {}).get('enabled', False):
            # The diagnostic measures by applied update, including mid-epoch.
            # Never silently change its installed classifier or reference here.
            return False
        validation_due = (epoch+1) % self.cfg['validation_every_epochs'] == 0
        if self.cfg.get('validation_relative_to_install', False):
            validation_due = epoch-self.reward.last_refit_epoch >= self.cfg['validation_every_epochs']
        if validation_due:
            if getattr(self, 'fixed_round', None) is not None:
                self.fixed_round.verify()
            self.evaluate(epoch, step)
            if getattr(self, 'fixed_round', None) is not None:
                self.fixed_round.verify()
                if self.rank == 0:
                    self.emit({'tau/fixed_round/integrity_verified': 1,
                               'tau/fixed_round/anchor_step': 500,
                               'tau/fixed_round/updates_since_anchor': step-500}, step)
        if self.cfg.get('fixed_general_round', False):
            return False
        refit = not final and (epoch+1) % self.cfg['refit_every_epochs'] == 0
        if self.cfg.get('refit_relative_to_install', False):
            refit = not final and epoch-self.reward.last_refit_epoch >= self.cfg['refit_every_epochs']
        if refit:
            self.refit(epoch, step)
        return refit
