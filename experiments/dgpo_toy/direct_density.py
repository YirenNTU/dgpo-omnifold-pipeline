"""Numerical DDIM endpoint densities for the small conditional toy only.

No learned estimator: float64 change of variables, exact analytic Jacobians,
and per-step inverse contraction. A sufficient global injectivity bound is
required. This is checked floating-point arithmetic, not interval arithmetic.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import math

import torch

try:
    from .conditional import Denoiser, alpha_sigma, ddim
    from .low_ess_recovery import invertibility_certificate
except ImportError:
    from conditional import Denoiser, alpha_sigma, ddim
    from low_ess_recovery import invertibility_certificate


class DensityValidityError(RuntimeError):
    """Do not use a single inverse branch as a certified density."""


@dataclass(frozen=True)
class DensitySettings:
    coefficient: float = 1.
    inverse_tolerance: float = 1e-11
    residual_tolerance: float = 1e-8
    inverse_iterations: int = 128
    minimum_singular: float = 1e-8
    chunk: int = 512
    structure_every: int = 250

    def __post_init__(self):
        if not math.isfinite(self.coefficient) or self.coefficient < 0:
            raise ValueError("coefficient must be finite and nonnegative")
        if min(self.inverse_tolerance, self.residual_tolerance, self.minimum_singular) <= 0:
            raise ValueError("Density tolerances must be positive")
        if min(self.inverse_iterations, self.chunk, self.structure_every) < 1:
            raise ValueError("Density budgets must be positive")


class DDIMDensity:
    def __init__(self, model, steps, settings=DensitySettings()):
        self.settings, self.steps = settings, steps
        self.model = copy.deepcopy(model).double().eval().requires_grad_(False)
        if any(not torch.isfinite(p).all() for p in self.model.parameters()):
            raise DensityValidityError("Nonfinite DDIM parameters")
        self.certificate = invertibility_certificate(self.model, steps)
        if not self.certificate["certified"]:
            raise DensityValidityError("Global injectivity not certified: " + str(self.certificate))
        self.d = self.model.network[-1].out_features
        self.eye = torch.eye(self.d, dtype=torch.float64)
        self.schedule = []
        for i in range(steps, 0, -1):
            t = torch.tensor(i/steps, dtype=torch.float64)
            a, s = alpha_sigma(t)
            ap, sp = alpha_sigma(t-1/steps)
            self.schedule.append((t, ap*a+sp*s, sp*a-ap*s, s))

    @torch.no_grad()
    def velocity_jacobian(self, x, c, t, sigma):
        tf = torch.stack([t, (math.pi*t).sin(), (math.pi*t).cos(),
                          (2*math.pi*t).sin(), (2*math.pi*t).cos()])
        h = torch.cat([x, c, tf.expand(len(x), -1)], -1)
        jac = None
        for layer in self.model.network:
            if isinstance(layer, torch.nn.Linear):
                h = torch.nn.functional.linear(h, layer.weight, layer.bias)
                # Keep derivative coordinates before feature coordinates so
                # later linear layers use one contiguous matrix multiply.
                jac = (layer.weight[:, :self.d].T.expand(len(x), -1, -1)
                       if jac is None else torch.nn.functional.linear(jac, layer.weight))
            else:  # exact architecture has already been checked by certificate
                sigmoid = h.sigmoid()
                derivative = sigmoid*(1+h*(1-sigmoid))
                jac = derivative[:, None, :]*jac
                h = torch.nn.functional.silu(h)
        return sigma*h, sigma*jac.transpose(-1, -2)

    @torch.no_grad()
    def forward_density(self, z, c):
        """Forward map and summed logdet; used also to validate inverse results."""
        x = z
        logdet = torch.zeros(len(x), dtype=torch.float64)
        full = self.eye.expand(len(x), -1, -1)
        for t, a, b, s in self.schedule:
            v, jac = self.velocity_jacobian(x, c, t, s)
            step_jac = a*self.eye+b*jac
            sign, ld = torch.linalg.slogdet(step_jac)
            if (sign <= 0).any() or not torch.isfinite(ld).all():
                raise DensityValidityError("Nonpositive or nonfinite DDIM Jacobian determinant")
            logdet += ld
            full = step_jac@full
            x = a*x+b*v
        return x, logdet, full

    @torch.no_grad()
    def inverse(self, y, c):
        """Reverse each Ax+Bv step by its certified contractive iteration."""
        x = y.clone()
        iterations_max, error_bound_max = 0, 0.
        lip = self.certificate["network_lipschitz_upper_bound"]
        for t, a, b, s in reversed(self.schedule):
            target = x
            guess = target/a
            contraction = float(b.abs()*s*lip/a)
            for iteration in range(1, self.settings.inverse_iterations+1):
                updated = (target-b*self.model(guess, t, c))/a
                # Banach a-posteriori distance from updated to unique step root.
                bound = contraction/(1-contraction)*float((updated-guess).abs().amax(-1).max())*math.sqrt(self.d)
                guess = updated
                if not torch.isfinite(guess).all() or not math.isfinite(bound):
                    raise DensityValidityError("Nonfinite DDIM inverse")
                if bound <= self.settings.inverse_tolerance:
                    break
            else:
                raise DensityValidityError("DDIM inverse did not meet contraction error tolerance")
            x = guess
            error_bound_max = max(error_bound_max, bound)
            iterations_max = max(iterations_max, iteration)
        return x, {"inverse_iterations_max": iterations_max, "inverse_step_error_bound_max": error_bound_max}

    @torch.no_grad()
    def log_prob(self, y, c):
        shape = y.shape[:-1]
        y, c = y.detach().double().reshape(-1, self.d), c.detach().double().reshape(-1, c.shape[-1])
        if len(c) != len(y) or len(y) == 0 or not torch.isfinite(y).all() or not torch.isfinite(c).all():
            raise DensityValidityError("Invalid endpoint/condition data")
        scores, diagnostics = [], {"inverse_iterations_max": 0, "inverse_step_error_bound_max": 0.,
                                  "forward_residual_max": 0., "minimum_map_singular": math.inf}
        for yy, cc in zip(y.split(self.settings.chunk), c.split(self.settings.chunk)):
            z, info = self.inverse(yy, cc)
            reconstructed, ld, jac = self.forward_density(z, cc)
            residual = float((reconstructed-yy).abs().max())
            singular = float(torch.linalg.svdvals(jac)[..., -1].min())
            score = -.5*(z.square()+math.log(2*math.pi)).sum(-1)-ld
            if (residual > self.settings.residual_tolerance or singular < self.settings.minimum_singular
                    or not torch.isfinite(score).all() or not math.isfinite(singular)):
                raise DensityValidityError(f"Density reconstruction/conditioning failed: residual={residual}, singular={singular}")
            scores.append(score)
            for key, value in info.items():
                diagnostics[key] = max(diagnostics[key], value)
            diagnostics["forward_residual_max"] = max(diagnostics["forward_residual_max"], residual)
            diagnostics["minimum_map_singular"] = min(diagnostics["minimum_map_singular"], singular)
        return torch.cat(scores).reshape(shape), diagnostics

    def validate_autodiff(self, c, z):
        """Independent full-DDIM autograd Jacobian vs analytic chain rule."""
        c, z = c.double(), z.double()
        y, ld, jac = self.forward_density(z, c)
        with torch.enable_grad():
            expected_jac = torch.vmap(torch.func.jacrev(
                lambda zz, cc: ddim(self.model, cc, zz, self.steps)))(z, c)
        expected = ddim(self.model, c, z, self.steps)
        values = {"forward_error": float((expected-y).abs().max()),
                  "jacobian_error": float((expected_jac-jac).abs().max()),
                  "logdet_error": float((torch.linalg.slogdet(expected_jac)[1]-ld).abs().max())}
        if max(values.values()) > 1e-8:
            raise DensityValidityError("Independent autodiff check failed: " + str(values))
        return values


class EndpointRatio:
    """Stateless density evaluation; NO additional trainable classifier/RNG."""
    def __init__(self, settings=DensitySettings()):
        self.settings, self.coefficient = settings, settings.coefficient
        self.reference, self.current = None, None
        self.policy_step = 0
        self.diagnostics = {}

    def refresh(self, model, reference, steps):
        if self.reference is None:
            self.reference = DDIMDensity(reference, steps, self.settings)
        self.current = DDIMDensity(model, steps, self.settings)

    def update(self, model, reference, data, cfg, step):
        if step != self.policy_step+1:
            raise ValueError("Density policy clock must advance exactly once")
        self.refresh(model, reference, cfg.ddim_steps)
        self.policy_step = step
        self.diagnostics = {}
        return {"density/current_minimum_step_margin": self.current.certificate["minimum_step_margin"],
                "density/current_maximum_inverse_contraction": self.current.certificate["maximum_inverse_step_contraction"],
                "density/reference_minimum_step_margin": self.reference.certificate["minimum_step_margin"]}

    def __call__(self, y, c, data=None):
        if self.current is None:
            raise RuntimeError("Refresh density before scoring")
        current, ci = self.current.log_prob(y, c)
        reference, ri = self.reference.log_prob(y, c)
        self.diagnostics = {"density/"+label+"_"+k: v
                            for label, info in (("current", ci), ("reference", ri)) for k, v in info.items()}
        # Evaluate BOTH densities at the same actual float32 rollout locations,
        # viewed in R^d. The modeled sampler is its float64 continuous DDIM map,
        # not a claim of a continuous density for discrete floating-point output.
        return (current-reference).to(y.dtype)

    def state_dict(self):
        return {"kind": "direct_ddim_density_v1", "settings": asdict(self.settings), "policy_step": self.policy_step}

    def load_state_dict(self, state):
        if state["kind"] != "direct_ddim_density_v1" or state["settings"] != asdict(self.settings):
            raise ValueError("Resume must preserve direct-density settings")
        self.policy_step = state["policy_step"]
        self.current, self.reference = None, None
