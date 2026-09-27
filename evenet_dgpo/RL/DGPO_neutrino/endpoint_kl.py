"""Online conditional endpoint reverse-KL estimator for deterministic DDIM.

Balanced logistic labels: current=1, frozen reference=0. The fitted logit
estimates log(q_current/q_reference). During the actor update its parameters
are frozen, but its input gradient is propagated through the FULL DDIM chain.
This is a finite-capacity plug-in estimate, not an exact likelihood or path KL.
Adding it to the existing DGPO surrogate does not make that surrogate an exact
gradient of the distribution-level reward-minus-KL objective.
"""
from __future__ import annotations

import copy
import math
import logging
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Mapping

import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F


CHECKPOINT_KEY = "dgpo_endpoint_kl_state"
_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EndpointKLConfig:
    enabled: bool = False
    coefficient: float = 1.0
    fit_steps: int = 20
    bootstrap_steps: int = 200
    fit_events_per_rank: int = 128
    validation_events_per_rank: int = 32
    actor_events_per_rank: int = 16
    microbatch_size: int = 8
    validation_interval: int = 10
    learning_rate: float = 2e-5
    backbone_learning_rate: float = 2e-6
    weight_decay: float = 1e-3
    grad_clip_norm: float = 5.0
    seed: int = 20260921

    @classmethod
    def parse(cls, raw=None):
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raw = vars(raw)
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown endpoint_kl settings: {sorted(unknown)}")
        result = cls(**raw)
        if type(result.enabled) is not bool:
            raise ValueError("endpoint_kl.enabled must be boolean")
        for name in ("fit_steps", "bootstrap_steps", "fit_events_per_rank",
                     "validation_events_per_rank", "actor_events_per_rank",
                     "microbatch_size", "validation_interval"):
            if type(getattr(result, name)) is not int or getattr(result, name) < 1:
                raise ValueError(f"endpoint_kl.{name} must be a positive integer")
        for name in ("coefficient", "learning_rate", "backbone_learning_rate", "grad_clip_norm"):
            value = getattr(result, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"endpoint_kl.{name} must be finite and positive; use enabled=false to switch off")
        if not math.isfinite(result.weight_decay) or result.weight_decay < 0:
            raise ValueError("endpoint_kl.weight_decay must be finite and nonnegative")
        if type(result.seed) is not int or result.seed < 0:
            raise ValueError("endpoint_kl.seed must be a nonnegative integer")
        return result


def validate_endpoint_protocol(cfg, dgpo, *, generation_uses_ema=False):
    """Validate before a long bootstrap fit, with the disabled path a no-op."""
    if not cfg.enabled:
        return
    adaptive = dgpo.get("adaptive_omnifold", {})
    fit = adaptive.get("recalibration", {})
    trigger = adaptive.get("trigger", {})
    if not adaptive.get("enabled", False) or int(fit.get("max_reward_rounds", 0)) != 1:
        raise ValueError("endpoint_kl requires a single frozen OmniFold reward round")
    if int(fit.get("max_iterations", 1)) != 1:
        raise ValueError("endpoint_kl requires iteration-one reward provenance")
    if generation_uses_ema:
        raise ValueError("endpoint_kl requires raw policy generation, not EMA")
    trust = dgpo.get("reference_trust", {})
    if trust.get("enabled", False) or trust.get("adaptive_boundary", {}).get("enabled", False):
        raise ValueError("endpoint_kl requires velocity/path reference trust and its boundary disabled")
    if float(dgpo.get("beta_kl", 0)) != 0:
        raise ValueError("endpoint_kl cannot be combined with the legacy denoising anchor")
    if dgpo.get("projection_constraint", {}).get("type", "none") != "none":
        raise ValueError("endpoint_kl requires projection_constraint.type=none")
    if trigger.get("rollback_to_best_on_plateau", False) or not adaptive.get("log_only", False):
        raise ValueError("endpoint_kl requires read-only adaptive monitoring, without policy rollback")
    if dgpo.get("gradient_transfer_trace", {}).get("enabled", False):
        raise ValueError("endpoint_kl is not included in the legacy two-component gradient_transfer trace")


@contextmanager
def evaluation_mode(model):
    # Preserve heterogeneous module modes, including frozen pretrained blocks.
    modules = list(model.modules())
    modes = [m.training for m in modules]
    model.eval()
    try:
        yield
    finally:
        for module, mode in zip(modules, modes):
            module.training = mode


def mean_across_ranks(value):
    value = value.detach().double().clone()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value)
        value /= dist.get_world_size()
    return value


def balanced_bce(current, reference):
    return (F.softplus(-current).mean() + F.softplus(reference).mean()) / 2


def select_batch(batch, indices):
    count = len(batch["x"])
    return {k: v.index_select(0, indices) if isinstance(v, torch.Tensor)
            and v.ndim and len(v) == count else v for k, v in batch.items()}


class EndpointKLController:
    """Not an nn.Module: critic never enters the policy optimizer/state_dict."""

    def __init__(self, config, saved=None):
        self.config = config
        self.classifier = None
        self.optimizer = None
        self.pending = saved
        self.updates = 0
        self.fit_updates = 0
        self.reference_round = None
        self.params = []
        self.last_metrics = {}
        if saved is not None:
            if saved.get("version") != 1:
                raise ValueError("unsupported endpoint KL checkpoint version")
            expected = asdict(config)
            # Coefficient and enabled are user switches, not critic architecture.
            old = dict(saved["config"])
            for key in ("enabled", "coefficient"):
                old.pop(key, None)
                expected.pop(key, None)
            if old != expected:
                raise ValueError("endpoint KL full resume requires matching critic settings; use weights_only for a new experiment")

    def initialize(self, source):
        if self.classifier is not None:
            if self.reference_round != source.reward_round_id:
                raise RuntimeError("endpoint KL reference/reward round changed")
            return
        self.classifier = source.make_endpoint_kl_classifier()
        self.packing_spec = self.classifier.packing_spec
        self.reference_round = source.reward_round_id
        named = [(n, p) for n, p in self.classifier.named_parameters() if p.requires_grad]
        self.params = [p for _, p in named]
        groups = []
        for backbone, lr in ((False, self.config.learning_rate), (True, self.config.backbone_learning_rate)):
            params = [p for n, p in named if n.startswith("backbone.") == backbone]
            if params:
                groups.append({"params": params, "lr": lr})
        self.optimizer = torch.optim.AdamW(groups, weight_decay=self.config.weight_decay)
        if self.pending and self.pending.get("classifier") is not None:
            saved = self.pending
            if saved["reference_round"] != self.reference_round:
                raise ValueError("saved endpoint KL critic belongs to a different reward/reference round")
            self.classifier.load_state_dict(saved["classifier"], strict=True)
            self.optimizer.load_state_dict(saved["optimizer"])
            self.updates = int(saved["updates"])
            self.fit_updates = int(saved["fit_updates"])
        self.pending = None
        if dist.is_available() and dist.is_initialized():
            state = self.classifier.state_dict()
            for tensor in state.values():
                dist.broadcast(tensor, src=0)
            self.classifier.load_state_dict(state, strict=True)

    def state_dict(self):
        if self.classifier is None and self.pending is not None:
            return self.pending
        return {"version": 1, "config": asdict(self.config),
                "classifier": None if self.classifier is None else self.classifier.state_dict(),
                "optimizer": None if self.optimizer is None else self.optimizer.state_dict(),
                "updates": self.updates, "fit_updates": self.fit_updates,
                "reference_round": self.reference_round}

    def _score(self, condition, candidates):
        return self.classifier(condition, candidates.flatten(1))

    @torch.no_grad()
    def _evaluate(self, condition, current, reference):
        self.classifier.eval()
        cur, ref = [], []
        for start in range(0, len(condition), self.config.microbatch_size):
            sl = slice(start, start + self.config.microbatch_size)
            cur.append(self._score(condition[sl], current[sl]))
            ref.append(self._score(condition[sl], reference[sl]))
        cur, ref = torch.cat(cur), torch.cat(ref)
        bce = mean_across_ranks(balanced_bce(cur, ref))
        acc = mean_across_ranks(((cur > 0).float().mean() + (ref <= 0).float().mean()) / 2)
        # E_ref exp(log q/ref)=1. Use logmeanexp without an overflow-prone exp.
        log_z = torch.logsumexp(ref.double(), 0) - math.log(len(ref))
        if dist.is_available() and dist.is_initialized():
            values = [torch.empty_like(log_z) for _ in range(dist.get_world_size())]
            dist.all_gather(values, log_z)
            log_z = torch.logsumexp(torch.stack(values), 0) - math.log(len(values))
        return bce, {"validation_bce": float(bce), "validation_balanced_accuracy": float(acc),
                     "validation_current_logit_mean": float(mean_across_ranks(cur.mean())),
                     "validation_reference_logit_mean": float(mean_across_ranks(ref.mean())),
                     "validation_log_mean_ratio_reference": float(log_z)}

    def backward(self, *, model, reference, source, batch, sampler, num_ddim_steps,
                 device, global_step, reduce_gradients, valid=None):
        """Add actor KL gradients to existing policy grads; synchronize outside."""
        from RL.DGPO_neutrino.sampling import generate_neutrino_candidates
        from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import pack_event_inputs

        start_time = time.monotonic()
        cfg = self.config
        if not cfg.enabled:
            return {}
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        seed = cfg.seed + int(global_step) * 100003 + rank * 1009
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        # KL sampling does not shift the baseline DGPO RNG stream.
        with torch.random.fork_rng(devices=devices), evaluation_mode(model), evaluation_mode(reference):
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed(seed)
            # A resumed controller rebuilds modules whereas an existing one
            # does not. Keep that construction from shifting sample RNG.
            with torch.random.fork_rng(devices=devices):
                self.initialize(source)
            eligible = torch.arange(len(batch["x"]), device=device)
            if valid is not None:
                eligible = eligible[valid.reshape(-1).bool()]
            total = cfg.fit_events_per_rank + cfg.validation_events_per_rank + cfg.actor_events_per_rank
            enough = torch.tensor(int(len(eligible) >= total), device=device)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(enough, op=dist.ReduceOp.MIN)
            if not enough.item():
                raise ValueError(f"endpoint KL needs at least {total} valid events per rank; lower event budgets or increase batch size")
            order = eligible[torch.randperm(len(eligible), device=device)[:total]]
            selected = select_batch(batch, order)
            # Shapes/masks are conditioning, truth values must never enter sampling.
            selected["x_invisible"] = torch.zeros_like(selected["x_invisible"])
            condition, _ = pack_event_inputs(selected, self.packing_spec)
            condition = condition.detach()
            count = cfg.fit_events_per_rank + cfg.validation_events_per_rank
            current, ref = [], []
            for a in range(0, count, cfg.microbatch_size):
                sub = select_batch(selected, torch.arange(a, min(a + cfg.microbatch_size, count), device=device))
                for policy, target in ((model, current), (reference, ref)):
                    target.append(generate_neutrino_candidates(policy, sub, sampler, K=1,
                        num_ddim_steps=num_ddim_steps, device=device)[0].detach())
            current, ref = torch.cat(current), torch.cat(ref)
            n = cfg.fit_events_per_rank
            val_args = (condition[n:count], current[n:], ref[n:])
            initial_bce, _ = self._evaluate(*val_args)
            best_bce = float(initial_bce)
            best_model = copy.deepcopy(self.classifier.state_dict())
            best_optimizer = copy.deepcopy(self.optimizer.state_dict())
            selected_step = 0
            steps = cfg.bootstrap_steps if self.updates == 0 else cfg.fit_steps
            last_loss = 0.0
            for step in range(1, steps + 1):
                self.classifier.train()
                self.optimizer.zero_grad(set_to_none=True)
                ids = torch.randperm(n, device=device)[:min(cfg.microbatch_size, n)]
                loss = balanced_bce(self._score(condition[ids], current[ids]), self._score(condition[ids], ref[ids]))
                finite = mean_across_ranks(torch.isfinite(loss).float())
                if float(finite) != 1.0:
                    raise FloatingPointError("nonfinite endpoint KL critic BCE")
                loss.backward()
                reduce_gradients(self.classifier)
                nn.utils.clip_grad_norm_(self.params, cfg.grad_clip_norm, error_if_nonfinite=True)
                self.optimizer.step()
                self.fit_updates += 1
                last_loss = float(mean_across_ranks(loss))
                if step % cfg.validation_interval == 0 or step == steps:
                    bce, _ = self._evaluate(*val_args)
                    if rank == 0:
                        _log.info("[DGPO/endpoint-KL] policy_step=%s fit=%s/%s train_bce=%.6g validation_bce=%.6g best=%.6g", global_step, step, steps, last_loss, float(bce), best_bce)
                    if float(bce) < best_bce:
                        best_bce, selected_step = float(bce), step
                        best_model = copy.deepcopy(self.classifier.state_dict())
                        best_optimizer = copy.deepcopy(self.optimizer.state_dict())
            self.classifier.load_state_dict(best_model, strict=True)
            self.optimizer.load_state_dict(best_optimizer)
            _, diagnostics = self._evaluate(*val_args)
            self.optimizer.zero_grad(set_to_none=True)
            policy_parameters = [p for p in model.parameters() if p.requires_grad]
            before = [torch.zeros_like(p) if p.grad is None else p.grad.detach().clone()
                      for p in policy_parameters]
            # Freeze parameters explicitly; no no_grad around classifier forward.
            for p in self.params:
                p.requires_grad_(False)
            actor_mean = torch.zeros((), device=device)
            actor_second = torch.zeros((), device=device)
            input_grad_norm = torch.zeros((), device=device)
            try:
                for a in range(count, total, cfg.microbatch_size):
                    b = min(a + cfg.microbatch_size, total)
                    sub = select_batch(selected, torch.arange(a, b, device=device))
                    draws = generate_neutrino_candidates(model, sub, sampler, K=1,
                        num_ddim_steps=num_ddim_steps, device=device,
                        differentiable=True, checkpoint_steps=True)[0]
                    score = self._score(condition[a:b], draws)
                    if not score.requires_grad:
                        raise RuntimeError("endpoint KL input gradient is disconnected")
                    input_gradient = torch.autograd.grad(score.sum(), draws, retain_graph=True)[0]
                    input_grad_norm += input_gradient.detach().flatten(1).norm(dim=1).sum() / cfg.actor_events_per_rank
                    loss = cfg.coefficient * score.sum() / cfg.actor_events_per_rank
                    if float(mean_across_ranks(torch.isfinite(loss).float())) != 1.0:
                        raise FloatingPointError("nonfinite endpoint KL actor loss")
                    loss.backward()
                    actor_mean += score.detach().sum() / cfg.actor_events_per_rank
                    actor_second += score.detach().square().sum() / cfg.actor_events_per_rank
            finally:
                for p in self.params:
                    p.requires_grad_(True)
            main_sq, kl_sq, dot = [torch.zeros((), device=device, dtype=torch.float64) for _ in range(3)]
            for parameter, main_grad in zip(policy_parameters, before):
                kl_grad = (torch.zeros_like(parameter) if parameter.grad is None else parameter.grad.detach()) - main_grad
                main_sq += main_grad.double().square().sum()
                kl_sq += kl_grad.double().square().sum()
                dot += (main_grad.double() * kl_grad.double()).sum()
            # These are means of per-rank norms/cosines BEFORE all-reduce;
            # deliberately not advertised as global gradient geometry.
            diagnostics["mean_rank_main_gradient_norm"] = float(mean_across_ranks(main_sq.sqrt()))
            diagnostics["mean_rank_kl_gradient_norm"] = float(mean_across_ranks(kl_sq.sqrt()))
            diagnostics["mean_rank_gradient_cosine"] = float(mean_across_ranks(dot / (main_sq * kl_sq).sqrt().clamp_min(1e-30)))
            del before
            actor_mean, actor_second = mean_across_ranks(actor_mean), mean_across_ranks(actor_second)
            diagnostics.update(estimate=float(actor_mean),
                actor_logit_std=float((actor_second - actor_mean.square()).clamp_min(0).sqrt()),
                weighted_loss=cfg.coefficient * float(actor_mean), coefficient=cfg.coefficient,
                input_gradient_norm=float(mean_across_ranks(input_grad_norm)),
                train_bce_last=last_loss, validation_bce_before_fit=float(initial_bce),
                selected_fit_step=selected_step, fit_steps=steps, fit_updates_total=self.fit_updates,
                fit_events_per_rank=n, validation_events_per_rank=cfg.validation_events_per_rank,
                actor_events_per_rank=cfg.actor_events_per_rank, reference_round=self.reference_round,
                num_ddim_steps=num_ddim_steps, policy_step=global_step,
                learning_rate=cfg.learning_rate, backbone_learning_rate=cfg.backbone_learning_rate,
                seconds=time.monotonic() - start_time, enabled=1.0)
            self.updates += 1
            self.last_metrics = {f"endpoint_kl/{k}": v for k, v in diagnostics.items()}
            return self.last_metrics
