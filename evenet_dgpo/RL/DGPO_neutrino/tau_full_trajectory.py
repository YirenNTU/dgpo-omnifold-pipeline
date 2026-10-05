"""Matched full-DDIM reward ablation; native optimizer/reference remain owners.

Both arms use the same chunked rollout and RNG recipe. Pathwise replays each
chunk with its exact noise/dropout state, backpropagates through all DDIM steps,
then adds that gradient to the native detached-candidate reference gradient.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
import time
import logging
import json
from pathlib import Path
import torch

from RL.DGPO_neutrino.sampling import generate_neutrino_candidates
from RL.DGPO_neutrino.tau_pathwise_reward import compute_reward_grad
from scripts.tau_gradient_balance import EMAGradientBalance, validate_reference_balance


def phase_clock(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)
    return time.perf_counter()


def record_phase(metrics, name, start, device):
    elapsed = phase_clock(device)-start
    metrics[f'trajectory/time/{name}_seconds'] = elapsed
    if torch.device(device).type == 'cuda':
        metrics[f'trajectory/memory/{name}_peak_allocated_mib'] = torch.cuda.max_memory_allocated(device)/(1024**2)
        metrics[f'trajectory/memory/{name}_peak_reserved_mib'] = torch.cuda.max_memory_reserved(device)/(1024**2)
    logging.getLogger(__name__).info('[trajectory timing] %s %.3fs (rank-local)', name, elapsed)


def slice_batch(batch, start, stop):
    size = len(batch['x'])
    return {key: value[start:stop] if torch.is_tensor(value) and value.ndim and len(value) == size
            else value for key, value in batch.items()}


def rng_state(device):
    return (torch.get_rng_state(), torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)


@contextmanager
def replay_rng(state, device):
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.set_rng_state(state[0])
        if state[1] is not None:
            torch.cuda.set_rng_state(state[1], device)
        yield


@contextmanager
def sampler_checkpoint_only(model):
    """DDIM already checkpoints each step; avoid nested PET block recompute.

    Keep this context active through backward, since checkpoint recomputation
    reads the module flags again. Reference and native forwards are unaffected.
    """
    modules = [(module, module.gradient_checkpointing) for module in model.modules()
               if hasattr(module, 'gradient_checkpointing')]
    try:
        for module, _ in modules:
            module.gradient_checkpointing = False
        yield
    finally:
        for module, enabled in modules:
            module.gradient_checkpointing = enabled


class FullTrajectoryController:
    def __init__(self, cfg):
        self.method = cfg['method']
        self.microbatch = cfg['event_microbatch']
        self.atol, self.rtol = cfg['parity_atol'], cfg['parity_rtol']
        coefficient = cfg.get('reference_coefficient', 1.0)
        if (isinstance(coefficient, bool) or not isinstance(coefficient, (int, float))
                or not math.isfinite(coefficient) or coefficient <= 0
                or (self.method != 'pathwise' and coefficient != 1.0)):
            raise ValueError('Positive finite reference coefficient required; native preserves coefficient 1')
        self.reference_coefficient = float(coefficient)
        from scripts.tau_classifier_kl import validate_classifier_kl, validate_classifier_trust
        self.classifier_kl_cfg = validate_classifier_kl(cfg.get('classifier_kl'))
        self.classifier_kl = None
        self.hard_trust_cfg = validate_classifier_trust(cfg.get('hard_trust_region'))
        self.hard_trust = None
        if self.hard_trust_cfg is not None and self.classifier_kl_cfg is None:
            raise ValueError('Hard classifier trust requires coefficient-one classifier KL')
        from scripts.tau_trajectory_reward_refit import validate_reward_refit
        fresh_refit = validate_reward_refit(cfg.get('reward_refit'))
        if self.classifier_kl_cfg is not None and (self.method != 'pathwise' or coefficient != 1.0
                or cfg.get('reference_balance') is not None or fresh_refit is None
                or fresh_refit['ratio_bound'] is not None):
            raise ValueError('Classifier KL requires pathwise, a fresh unbounded anchor reward and coefficient 1 without balancing')
        if fresh_refit is not None and (self.method != 'pathwise' or self.reference_coefficient != 1.0
                or cfg.get('reference_balance') is not None):
            raise ValueError('Fresh ratio-bound ablation requires pathwise, coefficient 1 and no dynamic balance')
        balance_cfg = validate_reference_balance(cfg.get('reference_balance'))
        if balance_cfg is not None and (self.method != 'pathwise' or coefficient != 1.0):
            raise ValueError('Dynamic reference balance requires pathwise with base reference coefficient 1')
        self.balance = EMAGradientBalance(balance_cfg) if balance_cfg is not None else None
        self._reference_grads = self._pending_balance = None
        if self.method not in ('native', 'pathwise') or type(self.microbatch) is not int or self.microbatch < 1:
            raise ValueError('Declare native/pathwise and a positive event microbatch')
        if any(not math.isfinite(v) or v <= 0 for v in (self.atol, self.rtol)):
            raise ValueError('Positive finite parity tolerances required')
        self.records = []
        self.metrics = {}

    @property
    def active(self):
        return self.method == 'pathwise'

    @property
    def native_reference_coefficient(self):
        return 0.0 if self.classifier_kl_cfg is not None else self.reference_coefficient

    def rollout(self, model, batch, sampler, *, K, num_ddim_steps, device):
        if K != 8 or num_ddim_steps != 20:
            raise ValueError('Full trajectory requires all K8 candidates and DDIM20')
        if any(isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and module.training
               for module in model.modules()):
            raise ValueError('Trajectory replay does not support training-mode BatchNorm updates')
        self.records = []
        self.metrics = {'trajectory/pathwise': float(self.active), 'trajectory/ddim_steps': 20.,
                        'trajectory/event_microbatch': float(self.microbatch),
                        'trajectory/reference_coefficient': self.reference_coefficient}
        phase_start = phase_clock(device)
        if torch.device(device).type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        chains = []
        for k in range(K):
            parts = []
            for start in range(0, len(batch['x']), self.microbatch):
                stop = min(len(batch['x']), start+self.microbatch)
                sub = slice_batch(batch, start, stop)
                state = rng_state(device)
                result = generate_neutrino_candidates(model, sub, sampler, K=1,
                    num_ddim_steps=num_ddim_steps, device=device, differentiable=False)[0]
                if not torch.isfinite(result).all():
                    raise ValueError('Full trajectory produced nonfinite candidates')
                self.records.append((k, start, stop, state))
                parts.append(result.detach())
            chains.append(torch.cat(parts))
        result = torch.stack(chains)
        record_phase(self.metrics, 'rollout', phase_start, device)
        return result

    @torch.no_grad()
    def score_rollout(self, aggregator, candidates, batch):
        """Use the backward pass's exact K1/event layout in both method arms.

        The original frozen reward implementation remains the value oracle.
        Matching the batch shape avoids comparing different GEMM kernels.
        """
        phase_start = phase_clock(candidates.device)
        totals, components = [], {}
        # Each frozen reward still forwards K1 chunks internally; grouping K
        # here reuses observed preprocessing without changing model batch shape.
        for start in range(0, len(batch['x']), self.microbatch):
            stop = min(len(batch['x']), start+self.microbatch)
            total, breakdown = aggregator.compute(candidates[:, start:stop], slice_batch(batch, start, stop))
            totals.append(total)
            for name, value in breakdown.items():
                components.setdefault(name, []).append(value)
        result = torch.cat(totals, dim=1), {name: torch.cat(values, dim=1) for name, values in components.items()}
        record_phase(self.metrics, 'reward_scoring', phase_start, candidates.device)
        return result

    def reward_failure(self, source, target, generated, sub, expected, score, *, k, start):
        """Diagnose on identical endpoints without weakening the parity gate."""
        with torch.no_grad():
            original_target = source.compute(target.detach(), sub)
            original_replay = source.compute(generated.detach(), sub)
        with torch.enable_grad():
            grad_target = compute_reward_grad(source, target.detach().requires_grad_(True), sub).detach()
        def error(a, b):
            return float((a.detach()-b.detach()).abs().max())
        details = dict(chain=k, event_start=start,
            reward_max_abs=error(score, expected), endpoint_max_abs=error(generated, target),
            original_batch_layout_max_abs=error(original_target, expected),
            fixed_endpoint_implementation_max_abs=error(grad_target, original_target),
            replay_endpoint_amplification_max_abs=error(original_replay, original_target),
            replay_implementation_max_abs=error(score, original_replay),
            atol=self.atol, rtol=self.rtol)
        folder = getattr(self, 'failure_directory', None)
        if folder is not None:
            import torch.distributed as dist
            rank = dist.get_rank() if dist.is_initialized() else 0
            folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
            path = folder / f'rank-{rank:02d}-chain-{k}-event-{start}.pt'
            payload = {name: value.detach().cpu() for name, value in sub.items() if torch.is_tensor(value)}
            torch.save(dict(details=details, batch=payload, target=target.detach().cpu(),
                replay=generated.detach().cpu(), expected=expected.detach().cpu(), score=score.detach().cpu(),
                original_target=original_target.cpu(), original_replay=original_replay.cpu(),
                grad_target=grad_target.cpu()), path)
            details['diagnostic_file'] = str(path)
        return ValueError('Differentiable reward differs from the original frozen reward: ' + json.dumps(details))

    def backward(self, model, batch, sampler, source, candidates, rewards, *, device):
        if not self.active:
            self.records.clear()
            return self.metrics
        phase_start = phase_clock(device)
        params = tuple(p for p in model.parameters() if p.requires_grad)
        # Keep the diagnostic snapshot on-device: avoid per-parameter PCIe copies.
        ref_grads = [None if p.grad is None else p.grad.detach().clone() for p in params]
        if self.balance is not None:
            if self._reference_grads is not None or self._pending_balance is not None:
                raise RuntimeError('Previous dynamic balance transaction was not committed')
            self._reference_grads = ref_grads
        loss_value, endpoint_error, reward_error, input_norm = [torch.zeros((), device=device, dtype=torch.float64) for _ in range(4)]
        kl_value, kl_input_norm, input_dot = [torch.zeros((), device=device, dtype=torch.float64) for _ in range(3)]
        kl_source = None
        if self.classifier_kl_cfg is not None:
            if self.classifier_kl is None or self.classifier_kl.source is None:
                raise ValueError('Classifier KL must be fitted before the actor backward')
            kl_source = self.classifier_kl.source
            if any(g is not None and torch.count_nonzero(g) for g in ref_grads):
                raise ValueError('Classifier KL must replace all native velocity/reference gradients')
        prepared_inputs = {}
        B, K = len(batch['x']), len(candidates)
        import torch.distributed as dist
        report_progress = not dist.is_initialized() or dist.get_rank() == 0
        logger = logging.getLogger(__name__)
        if report_progress:
            logger.info('[trajectory progress] pathwise backward starting: %d chunks', len(self.records))
        for chunk_index, (k, start, stop, state) in enumerate(self.records, 1):
            sub = slice_batch(batch, start, stop)
            if source is not None and hasattr(source, 'prepare_inputs'):
                if start not in prepared_inputs:
                    prepared_inputs[start] = source.prepare_inputs(sub)
                reward_kwargs = dict(prepared=prepared_inputs[start])
            else:
                reward_kwargs = {}
            with replay_rng(state, device), sampler_checkpoint_only(model):
                generated = generate_neutrino_candidates(model, sub, sampler, K=1,
                    num_ddim_steps=20, device=device, differentiable=True, checkpoint_steps=True)
                target = candidates[k:k+1, start:stop]
                if not torch.allclose(generated.detach(), target, atol=self.atol, rtol=self.rtol):
                    raise ValueError('Differentiable rollout differs from recorded noise/dropout replay')
                endpoint_error = torch.maximum(endpoint_error, (generated.detach()-target).abs().max())
                score = compute_reward_grad(source, generated, sub, **reward_kwargs)
                expected = rewards[k:k+1, start:stop]
                if not torch.allclose(score.detach(), expected, atol=self.atol, rtol=self.rtol):
                    raise self.reward_failure(source, target, generated, sub, expected, score, k=k, start=start)
                reward_error = torch.maximum(reward_error, (score.detach()-expected).abs().max())
                # Same all-event/all-K normalization; invalid events are zeroed
                # by the existing reward mask. Never divide by a chunk's length.
                loss = -score.sum() / (K*B)
                input_gradient, = torch.autograd.grad(loss, generated)
                if not torch.isfinite(input_gradient).all():
                    raise ValueError('Nonfinite reward derivative with respect to candidates')
                input_norm += input_gradient.detach().double().square().sum()
                if kl_source is not None:
                    # Same endpoints, masks, normalization, feature map and
                    # DDIM VJP. The KL head is frozen at this current policy.
                    kl_score = compute_reward_grad(kl_source, generated, sub, **reward_kwargs)
                    kl_loss = kl_score.sum() / (K*B)
                    kl_gradient, = torch.autograd.grad(kl_loss, generated)
                    if not torch.isfinite(kl_gradient).all():
                        raise FloatingPointError('Nonfinite classifier KL input gradient')
                    kl_value += kl_loss.detach().double()
                    kl_input_norm += kl_gradient.detach().double().square().sum()
                    input_dot += (input_gradient.detach().double()*kl_gradient.detach().double()).sum()
                    input_gradient = input_gradient + kl_gradient
                # The frozen classifier was already differentiated above.
                # Feed its exact endpoint VJP into the sampler once, avoiding
                # a second backward through the classifier and feature trunk.
                generated.backward(input_gradient.detach())
                loss_value += loss.detach()
            del generated, score, loss, input_gradient
            if report_progress and (chunk_index % 8 == 0 or chunk_index == len(self.records)):
                logger.info('[trajectory progress] backward chunks=%d/%d elapsed=%.1fs',
                    chunk_index, len(self.records), time.perf_counter()-phase_start)
        ref_sq, reward_sq, dot = [torch.zeros((), device=device, dtype=torch.float64) for _ in range(3)]
        finite_gradients = torch.ones((), device=device, dtype=torch.bool)
        for parameter, reference in zip(params, ref_grads, strict=True):
            if parameter.grad is None:
                continue
            total = parameter.grad.detach().double()
            finite_gradients &= torch.isfinite(total).all()
            reference = torch.zeros_like(total) if reference is None else reference.double()
            reward = total-reference
            ref_sq += reference.square().sum()
            reward_sq += reward.square().sum()
            dot += (reference*reward).sum()
        if not finite_gradients:
            raise ValueError('Nonfinite full trajectory parameter gradient')
        loss_value, endpoint_error, reward_error, input_norm, ref_sq, reward_sq, dot = torch.stack(
            [loss_value, endpoint_error, reward_error, input_norm, ref_sq, reward_sq, dot]).cpu().tolist()
        self.metrics.update({
            'trajectory/reward_loss': loss_value,
            'trajectory/endpoint_parity_max_abs': endpoint_error,
            'trajectory/reward_parity_max_abs': reward_error,
            'trajectory/local_reward_gradient_norm': math.sqrt(reward_sq),
            'trajectory/local_reward_gradient_zero': float(reward_sq == 0),
            'trajectory/local_reference_gradient_norm': math.sqrt(ref_sq),
            'trajectory/local_unweighted_reference_gradient_norm': math.sqrt(ref_sq)/self.reference_coefficient,
            'trajectory/local_reward_to_reference_gradient_ratio': math.sqrt(reward_sq)/max(math.sqrt(ref_sq), 1e-30),
            'trajectory/local_reference_gradient_zero': float(ref_sq == 0),
            'trajectory/local_total_on_reward_projection_ratio': 1.0+dot/max(reward_sq, 1e-30),
            'trajectory/local_reward_reference_cosine': dot/max(math.sqrt(ref_sq*reward_sq), 1e-30),
            'trajectory/local_reward_input_gradient_norm': math.sqrt(input_norm),
        })
        if kl_source is not None:
            kval, knorm, cross = torch.stack((kl_value, kl_input_norm, input_dot)).cpu().tolist()
            self.metrics.update({'trajectory/classifier_kl/loss': kval,
                'trajectory/classifier_kl/coefficient': 1.,
                'trajectory/classifier_kl/fit_policy_step': float(self.classifier_kl.fit_step),
                'trajectory/classifier_kl/velocity_penalty_enabled': 0.,
                'trajectory/classifier_kl/local_input_gradient_norm': math.sqrt(knorm),
                'trajectory/classifier_kl/local_reward_kl_input_cosine': cross/max(math.sqrt(input_norm*knorm), 1e-30),
                'trajectory/objective_loss': loss_value+kval,
                'trajectory/local_objective_gradient_norm': math.sqrt(reward_sq)})
            # The existing reference-gradient diagnostics describe velocity
            # gradients; do not mislabel the combined endpoint objective.
            for name in ('local_reward_gradient_norm', 'local_reward_gradient_zero',
                         'local_reference_gradient_norm', 'local_unweighted_reference_gradient_norm',
                         'local_reward_to_reference_gradient_ratio', 'local_reference_gradient_zero',
                         'local_total_on_reward_projection_ratio', 'local_reward_reference_cosine'):
                self.metrics.pop('trajectory/'+name, None)
        record_phase(self.metrics, 'pathwise_backward_and_diagnostics', phase_start, device)
        self.records.clear()
        return self.metrics

    def finalize_gradients(self, model, *, world_size, device):
        """After total-gradient averaging, average reference once and rescale it."""
        if self.balance is None:
            return self.metrics
        import torch.distributed as dist
        if self._reference_grads is None:
            raise RuntimeError('Missing reference snapshot for dynamic balance')
        start = phase_clock(device)
        params = tuple(p for p in model.parameters() if p.requires_grad)
        worker_device = torch.device(device)
        if (not params or len(params) != len(self._reference_grads)
                or any(p.device != params[0].device or p.dtype != params[0].dtype for p in params)
                or params[0].device.type != worker_device.type
                or (worker_device.index is not None and params[0].device.index != worker_device.index)):
            raise ValueError('Dynamic balance requires one worker device/dtype and stable trainable parameters')
        flat = torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                          for p, g in zip(params, self._reference_grads, strict=True)])
        self._reference_grads = None
        if world_size > 1:
            if not dist.is_initialized() or dist.get_world_size() != world_size:
                raise RuntimeError('Dynamic balance requires the declared distributed world')
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            flat.div_(world_size)
        flat.div_(self.reference_coefficient)
        ref_sq = torch.zeros((), device=device, dtype=torch.float64)
        reward_sq, dot = ref_sq.clone(), ref_sq.clone()
        offset = 0
        for p in params:
            reference = flat[offset:offset+p.numel()].view_as(p).double()
            total = torch.zeros_like(reference) if p.grad is None else p.grad.detach().double()
            reward = total-self.reference_coefficient*reference
            ref_sq += reference.square().sum()
            reward_sq += reward.square().sum()
            dot += (reference*reward).sum()
            offset += p.numel()
        ref_sq, reward_sq, dot = torch.stack([ref_sq, reward_sq, dot]).cpu().tolist()
        state, metrics = self.balance.propose(math.sqrt(reward_sq), math.sqrt(ref_sq))
        # Broadcast the tiny state, not local norm averages. Every rank uses the
        # same current-step coefficient and checkpointable EMA state.
        if world_size > 1:
            payload = torch.tensor([state['coefficient'], state['ema_reward'] or 0.,
                state['ema_reference'] or 0., state['steps']], device=device, dtype=torch.float64)
            dist.broadcast(payload, src=0)
            coefficient, ema_reward, ema_reference, steps = payload.cpu().tolist()
            state.update(coefficient=coefficient, ema_reward=ema_reward or None,
                         ema_reference=ema_reference or None, steps=int(steps))
            metrics.update(coefficient=coefficient, ema_reward_norm=ema_reward,
                           ema_reference_norm=ema_reference, steps=int(steps))
        coefficient = state['coefficient']
        offset = 0
        for p in params:
            reference = flat[offset:offset+p.numel()].view_as(p)
            if p.grad is not None:
                p.grad.add_(reference, alpha=coefficient-self.reference_coefficient)
            offset += p.numel()
        self._pending_balance = state
        self.metrics.update({f'trajectory/balance/{key}': value for key, value in metrics.items()})
        self.metrics.update({
            'trajectory/reference_coefficient': coefficient,
            'trajectory/global_reward_gradient_norm': math.sqrt(reward_sq),
            'trajectory/global_unweighted_reference_gradient_norm': math.sqrt(ref_sq),
            'trajectory/global_reference_gradient_norm': coefficient*math.sqrt(ref_sq),
            'trajectory/global_reward_to_reference_gradient_ratio': math.sqrt(reward_sq)/max(coefficient*math.sqrt(ref_sq), 1e-30),
            'trajectory/global_reward_reference_cosine': dot/max(math.sqrt(ref_sq*reward_sq), 1e-30),
            'trajectory/global_total_on_reward_projection_ratio': 1.+coefficient*dot/max(reward_sq, 1e-30),
            'trajectory/extra_reference_allreduces': float(world_size > 1),
        })
        record_phase(self.metrics, 'dynamic_balance', start, device)
        return self.metrics

    def commit_balance(self):
        if self.balance is not None:
            if self._pending_balance is None:
                raise RuntimeError('No dynamic balance transaction to commit')
            self.balance.load_state_dict(self._pending_balance)
            self._pending_balance = None

    def balance_state_dict(self):
        if self._pending_balance is not None:
            raise RuntimeError('Cannot checkpoint an uncommitted balance transaction')
        return self.balance.state_dict() if self.balance is not None else None
