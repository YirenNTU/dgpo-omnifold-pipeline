"""Local conditional neural-DDIM / learned-Fourier reward experiment.

See CONDITIONAL_PROTOCOL.md. No Ray, W&B, production edits, or GPU required.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

try:
    from .run import (alpha_sigma, auc, build_dgpo_loss, compute_advantage,
                      invert_endpoint_variance, build_reference_trust_loss)
except ImportError:
    from run import (alpha_sigma, auc, build_dgpo_loss, compute_advantage,
                     invert_endpoint_variance, build_reference_trust_loss)


@dataclass(frozen=True)
class Config:
    dimensions: int = 12
    context_dim: int = 3
    hidden: int = 64
    classifier_hidden: int = 128
    harmonics: int = 4
    kappa: float = 8.
    structured_mass: float = .9
    pretrain_steps: int = 5000
    classifier_steps: int = 5000
    policy_steps: int = 300
    fit_batch: int = 512
    train_events: int = 32768
    validation_events: int = 8192
    test_events: int = 16384
    eval_events: int = 4096
    batch: int = 64
    candidates: int = 8
    timesteps: int = 4
    ddim_steps: int = 20
    eval_every: int = 25
    fit_lr: float = 1e-3
    classifier_lr: float = 3e-4
    policy_lr: float = 1e-4
    weight_decay: float = .001
    standardize_step: int = 50


def generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


class Distribution:
    """Four triple-phase copulas, with exactly unchanged lower-order marginals."""
    def __init__(self, cfg: Config):
        if cfg.dimensions % 3 or cfg.dimensions < 3:
            raise ValueError("dimensions must be a positive multiple of three")
        self.cfg = cfg
        rng = generator(7103)
        self.mean_matrix = torch.randn(cfg.context_dim, cfg.dimensions, generator=rng) * .35
        self.phase_matrix = torch.randn(cfg.context_dim, cfg.dimensions // 3, generator=rng)

    def contexts(self, n: int, rng: torch.Generator) -> Tensor:
        return 2 * torch.rand(n, self.cfg.context_dim, generator=rng) - 1

    def mean(self, c: Tensor) -> Tensor:
        value = c @ self.mean_matrix.to(c)
        return value + .2 * value.sin()

    def phase(self, c: Tensor) -> Tensor:
        value = c @ self.phase_matrix.to(c)
        return .8 * value + .3 * value.sin()

    def angles(self, y: Tensor, c: Tensor) -> Tensor:
        # Per-coordinate marginal transform only; no triple identities exposed.
        z = y - self.mean(c)
        return math.pi * (1 + torch.erf(z / math.sqrt(2)))

    def sample(self, c: Tensor, rng: torch.Generator, *, truth: bool) -> Tensor:
        n, d = len(c), self.cfg.dimensions
        if not truth:
            return self.mean(c) + torch.randn(n, d, generator=rng)
        angles = 2 * math.pi * torch.rand(n, d // 3, 3, generator=rng)
        np_rng = np.random.default_rng(int(torch.randint(2**31, (), generator=rng)))
        phase = torch.from_numpy(np_rng.vonmises(0., self.cfg.kappa, (n, d // 3))).float()
        last = (phase + self.phase(c) - angles[..., :2].sum(-1)).remainder(2 * math.pi)
        structured = torch.rand(n, 1, generator=rng) < self.cfg.structured_mass
        angles[..., 2] = torch.where(structured, last, angles[..., 2])
        uniform = (angles.flatten(1) / (2 * math.pi)).clamp(1e-7, 1-1e-7)
        return self.mean(c) + math.sqrt(2) * torch.erfinv(2 * uniform - 1)

    def joint_signal(self, y: Tensor, c: Tensor) -> Tensor:
        phase = self.angles(y, c).reshape(*y.shape[:-1], -1, 3).sum(-1) - self.phase(c)
        return phase.cos().mean(-1)

    def nominal_log_ratio(self, y: Tensor, c: Tensor) -> Tensor:
        phase = self.angles(y, c).reshape(*y.shape[:-1], -1, 3).sum(-1) - self.phase(c)
        k = y.new_tensor(self.cfg.kappa)
        log_i0 = torch.special.i0e(k).log() + k
        log_h = (k * phase.cos() - log_i0).sum(-1)
        m = self.cfg.structured_mass
        return torch.logaddexp(log_h + math.log(m), torch.full_like(log_h, math.log1p(-m)))

    def nominal_ess(self) -> float:
        k = torch.tensor(self.cfg.kappa, dtype=torch.float64)
        log_second = torch.special.i0e(2*k).log() - 2*torch.special.i0e(k).log()
        m = self.cfg.structured_mass
        return float(1 / (1-m*m + m*m * (log_second * (self.cfg.dimensions // 3)).exp()))


class Denoiser(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(cfg.dimensions + cfg.context_dim + 5, cfg.hidden), nn.SiLU(),
            nn.Linear(cfg.hidden, cfg.hidden), nn.SiLU(),
            nn.Linear(cfg.hidden, cfg.hidden), nn.SiLU(),
            nn.Linear(cfg.hidden, cfg.dimensions),
        )

    def predict_features(self, x: Tensor, t: Tensor, c: Tensor):
        t = t.expand(x.shape[:-1])
        c = c.expand(*x.shape[:-1], c.shape[-1])
        time_features = torch.stack([t, (math.pi*t).sin(), (math.pi*t).cos(),
                                     (2*math.pi*t).sin(), (2*math.pi*t).cos()], -1)
        hidden = self.network[:-1](torch.cat([x, c, time_features], -1))
        return alpha_sigma(t)[1][..., None] * self.network[-1](hidden), hidden

    def forward(self, x: Tensor, t: Tensor, c: Tensor) -> Tensor:
        return self.predict_features(x, t, c)[0]


def ddim(model: Denoiser, c: Tensor, noise: Tensor, steps: int) -> Tensor:
    """Full differentiable DDIM chain. DGPO caller explicitly detaches it."""
    x = noise
    for i in range(steps, 0, -1):
        t = x.new_tensor(i / steps)
        a, s = alpha_sigma(t)
        ap, sp = alpha_sigma(t - 1 / steps)
        v = model(x, t, c)
        x = ap * (a*x-s*v) + sp * (s*x+a*v)
    return x


class Classifier(nn.Module):
    def __init__(self, cfg: Config, fourier: bool):
        super().__init__()
        self.fourier, self.harmonics = fourier, cfg.harmonics
        features = cfg.context_dim + cfg.dimensions * (1 + 2*cfg.harmonics*int(fourier))
        h = cfg.classifier_hidden
        self.encoder = nn.Sequential(nn.Linear(features, h), nn.GELU(),
                                     nn.Linear(h, h), nn.GELU(), nn.Linear(h, h), nn.GELU())
        self.head = nn.Linear(h, 1)
        self.register_buffer("center", torch.zeros(h))
        self.register_buffer("scale", torch.ones(h))

    def features(self, y: Tensor, c: Tensor, data: Distribution) -> Tensor:
        angles = data.angles(y, c)
        parts = [c, angles / math.pi - 1]
        if self.fourier:
            for k in range(1, self.harmonics + 1):
                parts.extend([(k*angles).sin(), (k*angles).cos()])
        return torch.cat(parts, -1)

    def forward(self, y: Tensor, c: Tensor, data: Distribution) -> Tensor:
        hidden = self.encoder(self.features(y, c, data))
        return self.head((hidden-self.center)/self.scale).squeeze(-1)

    @torch.no_grad()
    def standardize(self, y: Tensor, c: Tensor, data: Distribution) -> None:
        hidden = self.encoder(self.features(y, c, data))
        center, scale = hidden.mean(0), hidden.std(0, unbiased=False).clamp_min(1e-3)
        old_weight = self.head.weight.clone() / self.scale
        self.head.bias.add_((old_weight * (center-self.center)).sum(-1))
        self.head.weight.copy_(old_weight * scale)
        self.center.copy_(center)
        self.scale.copy_(scale)


def weight_health(logits: Tensor) -> dict:
    values = logits.detach().double().flatten()
    weights = values.softmax(0)
    return {"ess_fraction": float(1/(len(values)*weights.square().sum())),
            "log_mean_ratio": float(values.logsumexp(0)-math.log(len(values))),
            "top1pct_mass": float(weights.topk(max(1, math.ceil(.01*len(values)))).values.sum()),
            "max_weight_mass": float(weights.max()), "logit_max": float(values.max())}


def paired_gain(after: Tensor, before: Tensor) -> dict:
    # Each row is one independent context, columns are dependent candidates.
    difference = (after-before).double().mean(-1)
    gain = float(difference.mean())
    se = float(difference.std(unbiased=True) / math.sqrt(len(difference)))
    return {"gain": gain, "se": se, "lo95": gain-1.96*se, "hi95": gain+1.96*se}


def baseline_health(y: Tensor, c: Tensor, data: Distribution) -> dict:
    z = (y-data.mean(c)).double()
    n = len(z)
    covariance = (z-z.mean(0)).T @ (z-z.mean(0)) / n
    diagonal = covariance.diag()
    offdiag = covariance-torch.diag(diagonal)
    uniform = .5*(1+torch.erf(z.sort(0).values/math.sqrt(2)))
    high = torch.arange(1, n+1, dtype=z.dtype)[:, None]/n
    ks = torch.maximum(high-uniform, uniform-(high-1/n)).max()
    bin_errors = []
    for j in range(c.shape[-1]):
        for lo, hi in [(-1., -.5), (-.5, 0.), (0., .5), (.5, 1.0001)]:
            selected = z[(c[:, j] >= lo) & (c[:, j] < hi)]
            if len(selected):
                bin_errors.append(float(selected.mean(0).abs().max()))
    return {"max_marginal_ks": float(ks), "max_variance_error": float((diagonal-1).abs().max()),
            "max_offdiag_covariance": float(offdiag.abs().max()),
            "max_context_bin_mean_error": max(bin_errors)}


def initialize(cls, cfg: Config, seed: int, *args):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return cls(cfg, *args)


def pretrain(cfg: Config, data: Distribution, seed: int, emit):
    model = initialize(Denoiser, cfg, seed)
    rng = generator(seed+1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.fit_lr, weight_decay=cfg.weight_decay)
    nominal = float(invert_endpoint_variance(torch.ones(1, dtype=torch.float64), cfg.ddim_steps))
    variance = math.exp(nominal)
    history = []
    for step in range(1, cfg.pretrain_steps+1):
        c = data.contexts(cfg.fit_batch, rng)
        mean = data.mean(c)
        y = mean + math.sqrt(variance)*torch.randn(mean.shape, generator=rng)
        t = torch.rand(cfg.fit_batch, generator=rng)
        a, s = alpha_sigma(t[:, None])
        noisy = a*y + s*torch.randn(y.shape, generator=rng)
        teacher = a*s*(1-variance)/(a.square()*variance+s.square())*(noisy-a*mean)-s*mean
        loss = F.mse_loss(model(noisy, t, c), teacher)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step == 1 or step % 250 == 0 or step == cfg.pretrain_steps:
            row = {"phase": "pretrain_teacher", "step": step, "mse": float(loss.detach())}
            history.append(row)
            emit(row)
    return model.eval(), history


@torch.no_grad()
def make_panel(model, data, cfg, n, seed):
    rng = generator(seed)
    c = data.contexts(n, rng)
    truth = data.sample(c, rng, truth=True)
    noise = torch.randn(n, cfg.dimensions, generator=rng)
    samples = torch.cat([ddim(model, cc, zz, cfg.ddim_steps)
                         for cc, zz in zip(c.split(1024), noise.split(1024))])
    return {"context": c, "truth": truth, "generated": samples}


@torch.no_grad()
def scores(classifier, panel, data):
    result = []
    for name in ["truth", "generated"]:
        result.append(torch.cat([classifier(y, c, data) for y, c in
                                 zip(panel[name].split(1024), panel["context"].split(1024))]))
    return result


@torch.no_grad()
def classifier_metrics(classifier, panel, data):
    positive, negative = scores(classifier, panel, data)
    return {"bce": float(.5*(F.softplus(-positive).mean()+F.softplus(negative).mean())),
            "auc": auc(positive, negative), **weight_health(negative),
            "truth_reward_mean": float(positive.mean()), "generated_reward_mean": float(negative.mean())}


def fit_classifier(cfg, data, train, validation, seed, fourier, emit, *, model_class=Classifier):
    model = initialize(model_class, cfg, seed, fourier)
    rng = generator(seed+2000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.classifier_lr, weight_decay=cfg.weight_decay)
    best, best_state, best_step = math.inf, None, 0
    history = []
    label = "fourier" if fourier else "plain"
    for step in range(1, cfg.classifier_steps+1):
        indices = torch.randint(len(train["context"]), (cfg.fit_batch//2,), generator=rng)
        c = torch.cat([train["context"][indices]]*2)
        y = torch.cat([train["truth"][indices], train["generated"][indices]])
        target = torch.cat([torch.ones(len(indices)), torch.zeros(len(indices))])
        if step == cfg.standardize_step:
            count = min(2048, len(train["context"]))
            model.standardize(torch.cat([train["truth"][:count], train["generated"][:count]]),
                              train["context"][:count].repeat(2, 1), data)
            # Reparameterization invalidates old head moments, not encoder moments.
            for parameter in model.head.parameters():
                optimizer.state.pop(parameter, None)
        loss = F.binary_cross_entropy_with_logits(model(y, c, data), target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step % 100 == 0 or step == cfg.classifier_steps:
            values = classifier_metrics(model, validation, data)
            row = {"phase": "classifier", "classifier": label, "step": step,
                   "train_bce": float(loss.detach()), **values}
            history.append(row)
            emit(row)
            if values["bce"] < best:
                best, best_state, best_step = values["bce"], copy.deepcopy(model.state_dict()), step
    model.load_state_dict(best_state)
    model.eval().requires_grad_(False)
    return model, {"selected_step": best_step, "validation_bce": best,
                   "parameters": sum(p.numel() for p in model.parameters()), "history": history}


def head_group_gradients(error, hidden, advantage, gate, sigma):
    """Exact per-context last-layer gradients (before averaging contexts).

    This is a named head-only diagnostic, not whole-network influence/ESS.
    """
    m, k, _, d = error.shape
    derivative = (2/d) * error.detach() * advantage[None, ..., None]
    derivative = derivative * gate[:, None, :, None] * sigma[:, None, :, None]
    weight = torch.einsum("mkbd,mkbh->bdh", derivative, hidden.detach())/(m*k)
    bias = derivative.mean((0, 1))
    return torch.cat([weight.flatten(1), bias], -1)


def component_gradient_trace(main, penalty, parameters):
    """Pre-Adam/pre-clipping full-network vectors, without changing .grad or RNG."""
    parameters = tuple(parameters)
    g = torch.cat([x.detach().flatten() for x in torch.autograd.grad(
        main, parameters, retain_graph=True)]).double()
    h = torch.cat([x.detach().flatten() for x in torch.autograd.grad(
        penalty, parameters, retain_graph=True)]).double()
    ng, nh = float(g.norm()), float(h.norm())
    return {"main_gradient_norm": ng, "velocity_gradient_norm": nh,
            "velocity_to_main_gradient_ratio": nh/ng if ng > 0 else None,
            "main_velocity_gradient_cosine": float(g@h)/(ng*nh) if ng*nh > 0 else None,
            "total_on_main_projection": float(g@(g+h))/(ng*ng) if ng > 0 else None,
            "component_cancellation_ratio": float((g+h).norm())/(ng+nh) if ng+nh > 0 else None}


def dgpo_objective(model, reference, critic, data, cfg, c, noise, t, eps,
                   velocity_coefficient=0., trace_gradients=False,
                   endpoint_ratio=None, endpoint_coefficient=1., component_callback=None):
    if not math.isfinite(velocity_coefficient) or velocity_coefficient < 0:
        raise ValueError("velocity_coefficient must be finite and non-negative")
    if not math.isfinite(endpoint_coefficient) or endpoint_coefficient < 0:
        raise ValueError("endpoint_coefficient must be finite and non-negative")
    if endpoint_ratio is not None and velocity_coefficient:
        raise ValueError("Net endpoint reward and additive velocity penalty are separate ablations")
    b, k, d = noise.shape
    net_diagnostics = {}
    with torch.no_grad():
        candidates = ddim(model, c[:, None], noise, cfg.ddim_steps)
        reward = critic(candidates, c[:, None].expand(b, k, -1), data).T
        raw_reward = reward
        if endpoint_ratio is not None:
            kl_score = endpoint_ratio(candidates, c[:, None].expand(b, k, -1), data).T
            if kl_score.shape != reward.shape or not torch.isfinite(kl_score).all():
                raise ValueError("Endpoint ratio must return finite per-candidate scores")
            reward = raw_reward-endpoint_coefficient*kl_score
        advantage, _ = compute_advantage(reward, estimator="leave_one_out_unscaled")
        if endpoint_ratio is not None:
            raw_advantage, _ = compute_advantage(raw_reward, estimator="leave_one_out_unscaled")
            norm = raw_advantage.norm()*advantage.norm()
            net_diagnostics = {
                "net_reward_mean": float(reward.mean()),
                "net_reward_std": float(reward.std(unbiased=False)),
                "endpoint_kl_score_mean": float(kl_score.mean()),
                "endpoint_kl_score_std": float(kl_score.std(unbiased=False)),
                "endpoint_coefficient": endpoint_coefficient,
                "raw_advantage_abs_mean": float(raw_advantage.abs().mean()),
                "net_advantage_abs_mean": float(advantage.abs().mean()),
                "raw_net_advantage_cosine": float((raw_advantage*advantage).sum()/norm) if norm > 0 else None,
                "advantage_sign_flip_fraction": float(((raw_advantage*advantage)<0).float().mean()),
                "net_within_k_ess_fraction": float((1/(k*reward.softmax(0).square().sum(0))).mean()),
                **getattr(endpoint_ratio, "diagnostics", {}),
            }
        a, s = alpha_sigma(t[:, None, :, None])
        noisy = a*candidates.transpose(0, 1)[None] + s*eps[:, None]
        target = a*eps[:, None] - s*candidates.transpose(0, 1)[None]
        times = t[:, None].expand(-1, k, -1)
        contexts = c[None, None].expand(cfg.timesteps, k, b, -1)
        ref_v = reference(noisy, times, contexts)
        ref_loss = (ref_v-target).square().mean(-1)
    prediction, hidden = model.predict_features(noisy, times, contexts)
    error = prediction-target
    cur_loss = error.square().mean(-1)
    terms = [build_dgpo_loss(cur_loss[j], ref_loss[j], advantage, 1., k)[0]
             for j in range(cfg.timesteps)]
    gate = torch.sigmoid((advantage[None]*(cur_loss.detach()-ref_loss)).mean(1))
    group_grad = head_group_gradients(error, hidden, advantage, gate, alpha_sigma(t)[1])
    norms = group_grad.norm(dim=-1)
    total = float(norms.sum())
    contribution = norms/total if total > 0 else norms
    gradient_health = {
        "head_gradient_total_norm_sum": total,
        "head_gradient_ess_fraction": float(1/(b*contribution.square().sum())) if total > 0 else None,
        "head_gradient_top1pct_mass": float(contribution.topk(max(1, math.ceil(.01*b))).values.sum()) if total > 0 else None,
        "head_gradient_cancellation_ratio": float(group_grad.sum(0).norm()/norms.sum()) if total > 0 else None,
    }
    weights = raw_reward.softmax(0)
    loss = torch.stack(terms).mean()
    trust_diagnostics = {}
    if velocity_coefficient > 0:
        # All toy coordinates are active. Same detached candidates/t/eps as DGPO.
        trust, _ = build_reference_trust_loss(prediction, ref_v, torch.ones_like(prediction))
        weighted_trust = velocity_coefficient*trust
        if component_callback is not None:
            component_callback(loss, weighted_trust)
        trust_diagnostics = {"dgpo_loss": float(loss.detach()),
                             "velocity_mse": float(2*trust.detach()),
                             "velocity_penalty": float(weighted_trust.detach()),
                             "velocity_coefficient": velocity_coefficient}
        if trace_gradients:
            trust_diagnostics.update(component_gradient_trace(loss, weighted_trust, model.parameters()))
        loss = loss + weighted_trust
        trust_diagnostics["total_loss"] = float(loss.detach())
    return loss, {
        "rollout_reward": float(raw_reward.mean()), "rollout_reward_std": float(raw_reward.std(unbiased=False)),
        "informative_group_fraction": float(((reward.max(0).values-reward.min(0).values)>1e-3).float().mean()),
        "within_k_ess_fraction": float((1/(k*weights.square().sum(0))).mean()),
        "advantage_abs_mean": float(advantage.abs().mean()),
        "gate_mean": float(gate.mean()),
        "gate_saturated_fraction": float(((gate<.01)|(gate>.99)).float().mean()),
        **gradient_health,
        **trust_diagnostics,
        **net_diagnostics,
        **{"weight_"+key: value for key, value in weight_health(raw_reward).items()},
    }


@torch.no_grad()
def policy_panel(model, critic, data, cfg, seed):
    rng = generator(seed)
    c = data.contexts(cfg.eval_events, rng)
    noise = torch.randn(cfg.eval_events, cfg.candidates, cfg.dimensions, generator=rng)
    rewards, joints = [], []
    for cc, nn_ in zip(c.split(128), noise.split(128)):
        y = ddim(model, cc[:, None], nn_, cfg.ddim_steps)
        expanded = cc[:, None].expand(-1, cfg.candidates, -1)
        rewards.append(critic(y, expanded, data))
        joints.append(data.joint_signal(y, expanded))
    return torch.cat(rewards), torch.cat(joints)


def policy_train(arm, initial, critic, data, cfg, seed, monitor_seed, emit,
                 checkpoint_callback=None, resume_state=None, *,
                 velocity_coefficient=0., trace_gradients=True, endpoint_controller=None):
    if not math.isfinite(velocity_coefficient) or velocity_coefficient < 0:
        raise ValueError("velocity_coefficient must be finite and non-negative")
    if arm != "dgpo" and velocity_coefficient != 0:
        raise ValueError("Velocity penalty is implemented only for the DGPO arm")
    if resume_state is not None and resume_state.get("velocity_coefficient", 0.) != velocity_coefficient:
        raise ValueError("Resume must preserve velocity_coefficient")
    if endpoint_controller is not None and (arm != "dgpo" or velocity_coefficient):
        raise ValueError("Net endpoint reward requires DGPO without additive velocity penalty")
    if resume_state is not None:
        saved_endpoint = resume_state.get("endpoint_controller")
        if (saved_endpoint is None) != (endpoint_controller is None):
            raise ValueError("Resume must preserve endpoint controller")
        if endpoint_controller is not None:
            endpoint_controller.load_state_dict(saved_endpoint)
    model = copy.deepcopy(initial).requires_grad_(True)
    reference = copy.deepcopy(initial).requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.policy_lr, weight_decay=cfg.weight_decay)
    rng = generator(seed+3000)
    base_reward, _ = policy_panel(initial, critic, data, cfg, monitor_seed)
    history = []
    start_step = 0
    if resume_state is not None:
        start_step = int(resume_state["step"])
        if not 0 < start_step < cfg.policy_steps:
            raise ValueError("Resume step must precede the requested final step")
        model.load_state_dict(resume_state["model"], strict=True)
        optimizer.load_state_dict(copy.deepcopy(resume_state["optimizer"]))
        rng.set_state(resume_state["rng"])
        history = copy.deepcopy(resume_state["history"])
        if len(history) != start_step or history[-1]["step"] != start_step:
            raise ValueError("Resume history and step disagree")
    for step in range(start_step+1, cfg.policy_steps+1):
        endpoint_diagnostics = (endpoint_controller.update(model, reference, data, cfg, step)
                                if endpoint_controller is not None else {})
        c = data.contexts(cfg.batch, rng)
        noise = torch.randn(cfg.batch, cfg.candidates, cfg.dimensions, generator=rng)
        t = .7*torch.rand(cfg.timesteps, cfg.batch, generator=rng)
        eps = torch.randn(cfg.timesteps, cfg.batch, cfg.dimensions, generator=rng)
        if arm == "dgpo":
            loss, diagnostics = dgpo_objective(
                model, reference, critic, data, cfg, c, noise, t, eps,
                velocity_coefficient=velocity_coefficient,
                endpoint_ratio=endpoint_controller,
                endpoint_coefficient=endpoint_controller.coefficient if endpoint_controller is not None else 1.,
                trace_gradients=trace_gradients and (
                    step == 1 or step % cfg.eval_every == 0 or step == cfg.policy_steps))
        elif arm == "pathwise":
            y = ddim(model, c[:, None], noise, cfg.ddim_steps)
            reward = critic(y, c[:, None].expand(-1, cfg.candidates, -1), data)
            loss = -reward.mean()
            diagnostics = {"rollout_reward": float(reward.detach().mean())}
        else:
            raise ValueError(arm)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
        before = [p.detach().clone() for p in model.parameters()]
        optimizer.step()
        displacement = math.sqrt(sum(float((p.detach()-old).square().sum())
                                     for p, old in zip(model.parameters(), before)))
        row = {"phase": "policy", "arm": arm, "step": step,
               "gradient_norm": grad_norm, "grad_clipped": grad_norm>1,
               "update_l2": displacement, **diagnostics, **endpoint_diagnostics}
        if step == 1 or step % cfg.eval_every == 0 or step == cfg.policy_steps:
            reward, joint = policy_panel(model, critic, data, cfg, monitor_seed)
            row.update({"monitor_reward": float(reward.mean()),
                        "monitor_reward_std": float(reward.std(unbiased=False)),
                        "monitor_joint_signal": float(joint.mean()),
                        **{"monitor_"+key: value for key, value in paired_gain(reward, base_reward).items()},
                        **{"monitor_weight_"+key: value for key, value in weight_health(reward).items()}})
            emit(row)
        history.append(row)
        if checkpoint_callback is not None:
            checkpoint_callback(step, model, optimizer, rng, history)
    return model.eval(), history


def validity(baseline, classifier, informative, joint_gap):
    return {"marginals": baseline["max_marginal_ks"] <= .06,
            "variance": baseline["max_variance_error"] <= .15,
            "pair_covariance": baseline["max_offdiag_covariance"] <= .08,
            "conditional_mean": baseline["max_context_bin_mean_error"] <= .10,
            "classifier_bce": classifier["bce"] <= .60,
            "classifier_auc": classifier["auc"] >= .70,
            "low_ess": classifier["ess_fraction"] <= .01,
            "ratio_normalization": abs(classifier["log_mean_ratio"]) <= .50,
            "within_context_signal": informative >= .01,
            "missing_joint_structure": joint_gap >= .20}


def decide(gates, endpoints, smoke=False):
    if smoke:
        return "smoke_only"
    if not all(gates.values()):
        return "inconclusive_setup"
    improved = lambda arm: endpoints[arm]["gain"] >= .10 and endpoints[arm]["lo95"] > 0
    if improved("dgpo"):
        return "reward_improves"
    return "dgpo_transfer_deficit" if improved("pathwise") else "inconclusive_actionability"


def run_seed(cfg, seed, output, smoke=False):
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    log = (output/"progress.jsonl").open("w")
    def emit(row):
        payload = {"seed": seed, **row}
        line = json.dumps(payload, allow_nan=False)
        print(line, flush=True)
        log.write(line+"\n")
        log.flush()
    try:
        data = Distribution(cfg)
        initial, pretraining = pretrain(cfg, data, seed, emit)
        panels = {name: make_panel(initial, data, cfg, count, seed+offset) for name, count, offset in
                  [("train", cfg.train_events, 10000), ("validation", cfg.validation_events, 20000),
                   ("test", cfg.test_events, 30000)]}
        classifiers, fits = {}, {}
        for name, fourier in [("plain", False), ("fourier", True)]:
            classifier, fit = fit_classifier(cfg, data, panels["train"], panels["validation"],
                                            seed+4000, fourier, emit)
            fit["test"] = classifier_metrics(classifier, panels["test"], data)
            classifiers[name], fits[name] = classifier, fit
            emit({"phase": "classifier_test", "classifier": name, **fit["test"]})
        test = panels["test"]
        baseline = baseline_health(test["generated"], test["context"], data)
        joint_gap = float(data.joint_signal(test["truth"], test["context"]).mean()
                          - data.joint_signal(test["generated"], test["context"]).mean())
        critic = classifiers["fourier"]
        reward, _ = policy_panel(initial, critic, data, cfg, seed+40000)
        informative = float(((reward.max(1).values-reward.min(1).values)>1e-3).float().mean())
        gates = validity(baseline, fits["fourier"]["test"], informative, joint_gap)
        report = {"config": asdict(cfg), "seed": seed, "smoke": smoke,
                  "denoiser_parameters": sum(p.numel() for p in initial.parameters()),
                  "pretraining": pretraining, "classifiers": fits, "baseline": baseline,
                  "joint_gap": joint_gap, "nominal_oracle_population_ess": data.nominal_ess(),
                  "nominal_oracle_empirical_weights": weight_health(data.nominal_log_ratio(
                      test["generated"], test["context"])),
                  "validity": gates, "endpoints": {}, "policy_histories": {}}
        emit({"phase": "setup_validity", "gates": gates, "baseline": baseline,
              "joint_gap": joint_gap, "nominal_ess": data.nominal_ess()})
        torch.save({"config": asdict(cfg), "initial": initial.state_dict(),
                    "classifiers": {key: val.state_dict() for key, val in classifiers.items()}}, output/"models.pt")
        if all(gates.values()) or smoke:
            endpoint_initial, _ = policy_panel(initial, critic, data, cfg, seed+60000)
            for arm in ["dgpo", "pathwise"]:
                model, history = policy_train(arm, initial, critic, data, cfg, seed, seed+50000, emit)
                endpoint, _ = policy_panel(model, critic, data, cfg, seed+60000)
                report["endpoints"][arm] = paired_gain(endpoint, endpoint_initial)
                report["policy_histories"][arm] = history
                torch.save(model.state_dict(), output/(arm+".pt"))
        report["decision"] = decide(gates, report["endpoints"], smoke)
        report["seconds"] = time.perf_counter()-started
        (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        emit({"phase": "complete", "decision": report["decision"],
              "endpoints": report["endpoints"], "seconds": report["seconds"]})
        return report
    except BaseException as exc:
        (output/"failure.json").write_text(json.dumps({"status": "interrupted_or_error",
            "type": type(exc).__name__, "message": str(exc)}, indent=2)+"\n")
        raise
    finally:
        log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17])
    parser.add_argument("--classifier-steps", type=int, default=Config.classifier_steps,
                        help="Explicit new fit-budget variant; recorded in report config")
    parser.add_argument("--smoke", action="store_true", help="Execution test only; no scientific conclusion")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.classifier_steps < 1:
        parser.error("--classifier-steps must be positive")
    cfg = replace(Config(), classifier_steps=args.classifier_steps)
    if args.smoke:
        cfg = replace(cfg, hidden=24, classifier_hidden=24, pretrain_steps=20,
                      classifier_steps=20, policy_steps=3, fit_batch=64,
                      train_events=256, validation_events=128, test_events=256,
                      eval_events=32, batch=4, timesteps=2, eval_every=3, standardize_step=5)
    args.output.mkdir(parents=True, exist_ok=True)
    results = [run_seed(cfg, seed, args.output/f"seed{seed}", args.smoke) for seed in args.seeds]
    summary = {"protocol": "CONDITIONAL_PROTOCOL.md", "smoke": args.smoke,
               "results": [{"seed": r["seed"], "decision": r["decision"],
                            "seconds": r["seconds"], "validity": r["validity"],
                            "endpoints": r["endpoints"]} for r in results]}
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
