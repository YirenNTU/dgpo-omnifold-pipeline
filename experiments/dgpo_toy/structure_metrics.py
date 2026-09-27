"""Evaluation-only higher-order moments; no truth features enter policy loss."""
import itertools
import math

import numpy as np
import torch

try:
    from .conditional import ddim, generator
except ImportError:
    from conditional import ddim, generator


def structure_features(y, c, data, harmonics=4):
    groups = data.cfg.dimensions//3
    phase = data.angles(y, c).reshape(*y.shape[:-1], groups, 3).sum(-1)-data.phase(c)
    h = torch.arange(1, harmonics+1, device=y.device, dtype=y.dtype)
    single = phase[..., None]*h
    features = [single.cos().flatten(-2), single.sin().flatten(-2)]
    # Cross-triple features probe six-coordinate correlations, not only triples.
    for i, j in itertools.combinations(range(groups), 2):
        for sign in (-1, 1):
            angle = phase[..., i]+sign*phase[..., j]
            features.extend([angle.cos()[..., None], angle.sin()[..., None]])
    return torch.cat(features, -1)


def structure_target(data, harmonics=4):
    """Truth moment: mixture mass * I_h(kappa)/I_0(kappa), sin moments zero.

    Periodic quadrature avoids a scipy dependency. Shared mixture membership
    means cross-triple target is mass*rho1^2, NOT (mass*rho1)^2.
    """
    phase = (np.arange(65536)+.5)*(2*np.pi/65536)
    density = np.exp(data.cfg.kappa*(np.cos(phase)-1))
    rho = np.array([(density*np.cos(h*phase)).mean()/density.mean()
                    for h in range(1, harmonics+1)])
    groups, mass = data.cfg.dimensions//3, data.cfg.structured_mass
    values = [np.tile(mass*rho, groups), np.zeros(groups*harmonics)]
    for _ in itertools.combinations(range(groups), 2):
        for _ in (-1, 1):
            values.append(np.array([mass*rho[0]**2, 0.]))
    return np.concatenate(values)


@torch.no_grad()
def structure_panel(model, data, cfg, seed):
    rng = generator(seed)
    c = data.contexts(cfg.eval_events, rng)
    noise = torch.randn(cfg.eval_events, cfg.candidates, cfg.dimensions, generator=rng)
    features, residuals = [], []
    for cc, z in zip(c.split(128), noise.split(128)):
        y = ddim(model, cc[:, None], z, cfg.ddim_steps)
        context = cc[:, None].expand(-1, cfg.candidates, -1)
        features.append(structure_features(y, context, data).mean(1))
        residuals.append((y-data.mean(context)).flatten(0, 1))
    by_context = torch.cat(features).double().numpy()
    z = torch.cat(residuals).double()
    mean = z.mean(0)
    covariance = (z-mean).T@(z-mean)/len(z)
    offdiag = covariance-torch.diag(covariance.diag())
    target = structure_target(data)
    error = by_context.mean(0)-target
    triple_count = 2*(cfg.dimensions//3)*4
    summary = {"fourier_moment_rmse": float(np.sqrt(np.mean(error**2))),
               "triple_harmonic_rmse": float(np.sqrt(np.mean(error[:triple_count]**2))),
               "cross_triple_rmse": float(np.sqrt(np.mean(error[triple_count:]**2))) if len(error)>triple_count else None,
               "moment_mean": by_context.mean(0).tolist(), "truth_moment": target.tolist(),
               "marginal_mean_absmax": float(mean.abs().max()),
               "marginal_variance_error_absmax": float((covariance.diag()-1).abs().max()),
               "pair_covariance_absmax": float(offdiag.abs().max())}
    return by_context, summary


def paired_structure_change(after, before, target, seed, replicates=500):
    """Paired context bootstrap; negative MSE difference means closer to truth."""
    if after.shape != before.shape or len(after) < 2:
        raise ValueError("Matched structure panels with at least two contexts required")
    if not all(np.isfinite(a).all() for a in (after, before, target)):
        raise FloatingPointError("Nonfinite structure panel")
    def mse(a):
        return np.mean((a.mean(0)-target)**2)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(replicates):
        index = rng.integers(len(after), size=len(after))
        draws.append(mse(after[index])-mse(before[index]))
    lo, hi = np.quantile(draws, [.025, .975])
    return {"mse_change": float(mse(after)-mse(before)), "lo95": float(lo),
            "hi95": float(hi), "bootstrap_replicates": replicates,
            "decision": "closer_to_truth" if hi < 0 else "farther_from_truth" if lo > 0 else "unresolved"}
