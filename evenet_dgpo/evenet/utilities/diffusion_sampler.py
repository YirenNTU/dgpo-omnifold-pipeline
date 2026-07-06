import math

import torch
from torch import Tensor
from typing import Callable, Optional, Union
from tqdm import tqdm

from evenet.utilities.debug_tool import time_decorator


def logsnr_schedule_cosine(time: Tensor, logsnr_min: float = -20., logsnr_max: float = 20.) -> Tensor:
    logsnr_min = Tensor([logsnr_min]).to(time.device)
    logsnr_max = Tensor([logsnr_max]).to(time.device)
    b = torch.atan(torch.exp(-0.5 * logsnr_max)).to(time.device)
    a = (torch.atan(torch.exp(-0.5 * logsnr_min)) - b).to(time.device)
    return -2.0 * torch.log(torch.tan(a * time.to(torch.float32) + b))


def get_logsnr_alpha_sigma(time: Tensor, shape=None):
    logsnr = logsnr_schedule_cosine(time)
    alpha = torch.sqrt(torch.sigmoid(logsnr))
    sigma = torch.sqrt(torch.sigmoid(-logsnr))

    if shape is not None:
        logsnr = logsnr.view(shape).to(torch.float32)
        alpha = alpha.view(shape).to(torch.float32)
        sigma = sigma.view(shape).to(torch.float32)

    return logsnr, alpha, sigma


def ddim_time_at_step(time_step: int, num_steps: int) -> float:
    """Continuous diffusion time ``t`` at a DDIM iteration (cosine schedule input)."""
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    ts = int(time_step)
    if ts < 1 or ts > num_steps:
        raise ValueError(f"time_step must be in [1, {num_steps}], got {ts}")
    return float(ts) / float(num_steps)


def ddim_step_from_time(t: float, num_steps: int) -> int:
    """Nearest DDIM ``time_step`` for cosine-schedule time ``t = time_step / num_steps``."""
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    return max(1, min(num_steps, int(round(float(t) * num_steps))))


def ddim_visited_time_grid(
    num_steps: int,
    *,
    t_min: float = 0.0,
    t_max: float = 1.0,
    descending: bool = True,
) -> list[float]:
    """``t`` values on the uniform DDIM grid ``{k/num_steps}`` inside ``[t_min, t_max]``.

    These are the exact times passed to ``logsnr_schedule_cosine`` during rollout.
    """
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    lo = float(t_min)
    hi = float(t_max)
    if hi < lo:
        raise ValueError(f"t_max must be >= t_min, got t_min={lo}, t_max={hi}")
    grid: list[float] = []
    for time_step in range(1, num_steps + 1):
        t = ddim_time_at_step(time_step, num_steps)
        if lo - 1e-12 <= t <= hi + 1e-12:
            grid.append(t)
    if descending:
        grid.sort(reverse=True)
    else:
        grid.sort()
    return grid


def ddim_stratified_capture_t_fracs(
    num_steps: int,
    count: int,
    *,
    t_min: float = 0.0,
    t_max: float = 0.7,
) -> tuple[float, ...]:
    """Pick ``count`` stratified DDIM-grid ``t`` values in ``[t_min, t_max]`` (descending).

    Matches the discrete times the cosine-schedule rollout actually visits, not off-grid
    continuous midpoints.
    """
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")
    grid = ddim_visited_time_grid(num_steps, t_min=t_min, t_max=t_max, descending=True)
    if not grid:
        raise ValueError(
            f"no DDIM grid times in [t_min={t_min}, t_max={t_max}] for num_steps={num_steps}"
        )
    if count >= len(grid):
        return tuple(grid)
    selected: list[float] = []
    seen: set[float] = set()
    for i in range(count):
        j = int((i + 0.5) / float(count) * len(grid))
        j = min(j, len(grid) - 1)
        t = grid[j]
        if t in seen:
            continue
        seen.add(t)
        selected.append(t)
    if len(selected) < count:
        for t in grid:
            if t in seen:
                continue
            seen.add(t)
            selected.append(t)
            if len(selected) >= count:
                break
    return tuple(selected)


def add_noise(x: Tensor, time: Tensor) -> tuple[Tensor, Tensor]:
    """
    x: input tensor,
    time: time tensor (B,)
    """
    eps = torch.randn_like(x)
    time_expanded = time.view(time.shape[0], *([1] * (x.dim() - 1)))

    logsnr, alpha, sigma = get_logsnr_alpha_sigma(time_expanded)  # (B, 1, ...)
    perturbed_x = x * alpha + eps * sigma
    score = eps * alpha - x * sigma

    return perturbed_x, score


class DDIMSampler:
    def __init__(self, device):
        self.device = device

    def prior_sde(self, dimensions) -> Tensor:
        return torch.randn(dimensions, dtype=torch.float32, device=self.device)

    @time_decorator(name="DDIM sampler")
    def sample(
            self,
            data_shape,
            pred_fn: Callable,
            normalize_fn: Optional[torch.nn.Module] = None,
            num_steps: int = 20,
            eta: float = 1.0,
            noise_mask: Optional[torch.Tensor] = None,
            use_tqdm: bool = False,
            process_name: str = "Sampling",
            remove_padding: bool = False,
            capture_trajectory_states: bool = False,
            trajectory_capture_time_steps: Optional[list[int]] = None,
    ) -> Union[Tensor, tuple[Tensor, list[tuple[Tensor, Tensor]]]]:
        """
        DDIM sampling for diffusion models.
        
        Args:
            data_shape: Shape of the data to generate
            pred_fn: Model prediction function
            normalize_fn: Optional denormalization function
            num_steps: Number of sampling steps
            eta: Stochasticity parameter
            noise_mask: Optional mask for noise
            use_tqdm: Whether to show progress bar
            process_name: Name for progress bar
            remove_padding: Whether to remove padding in denormalization
            capture_trajectory_states: When True, also return selected normalized
                intermediate ``(x_t, t)`` pairs before final denormalization.
            trajectory_capture_time_steps: DDIM ``time_step`` indices in
                ``[1, num_steps]`` at which to snapshot ``x_t`` at ``t=time_step/num_steps``.
        """
        batch_size = data_shape[0]
        const_shape = (batch_size, *([1] * (len(data_shape) - 1)))
        x = self.prior_sde(data_shape)
        if noise_mask is not None:
            x = x * noise_mask

        capture_steps: set[int] = set()
        trajectory: list[tuple[Tensor, Tensor]] = []
        if capture_trajectory_states:
            if trajectory_capture_time_steps:
                capture_steps = {
                    max(1, min(num_steps, int(ts))) for ts in trajectory_capture_time_steps
                }
            else:
                capture_steps = {max(1, num_steps // 2)}

        iterable = range(num_steps, 0, -1)
        if use_tqdm:
            iterable = tqdm(iterable, desc=process_name, total=num_steps)

        for time_step in iterable:
            t = torch.ones((batch_size,)).to(self.device) * time_step / num_steps
            t = t.float()  # Convert to float if needed
            if capture_trajectory_states and time_step in capture_steps:
                trajectory.append((x.detach().clone(), t.detach().clone()))
            logsnr, alpha, sigma = get_logsnr_alpha_sigma(t, shape=const_shape)

            t_prev = torch.ones((batch_size,), device=self.device) * (time_step - 1) / num_steps
            t_prev = t_prev.float()
            logsnr_, alpha_, sigma_ = get_logsnr_alpha_sigma(t_prev, shape=const_shape)

            with torch.no_grad():
                output = pred_fn(noise_x=x, time=t)
                if not torch.isfinite(output).all():
                    raise RuntimeError(
                        f"[DDIM] non-finite model output at time_step={time_step}/{num_steps} "
                        f"(t={float(t[0]):.4f}); shape={tuple(output.shape)} "
                        f"nan_count={int((~torch.isfinite(output)).sum())}"
                    )

                # Velocity prediction: model outputs v
                v = output
                eps = v * alpha + x * sigma
                pred_x0 = (x - sigma * eps) / alpha.clamp(min=1e-8)

            # Standard DDIM update
            x = alpha_ * pred_x0 + sigma_ * eps
            if noise_mask is not None:
                x = x * noise_mask
            if not torch.isfinite(x).all():
                raise RuntimeError(
                    f"[DDIM] non-finite x after step time_step={time_step}/{num_steps} "
                    f"(t={float(t[0]):.4f}, alpha={float(alpha.flatten()[0]):.3e}, "
                    f"sigma={float(sigma.flatten()[0]):.3e}); "
                    f"nan_count={int((~torch.isfinite(x)).sum())}"
                )

        if normalize_fn is not None:
            x = normalize_fn.denormalize(x, noise_mask, remove_padding=remove_padding)
        if capture_trajectory_states:
            return x, trajectory
        return x

    @time_decorator(name="DDIM sampler (with log prob)")
    def sample_with_log_prob(
        self,
        data_shape,
        pred_fn: Callable,
        normalize_fn: Optional[torch.nn.Module] = None,
        num_steps: int = 100,
        eta: float = 1.0,
        noise_mask: Optional[torch.Tensor] = None,
        use_tqdm: bool = False,
        process_name: str = "Sampling",
        remove_padding: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """
        DDIM sampling with log p(x0) via change-of-variables (discrete).
        Model must predict velocity v_t = alpha_t * eps - sigma_t * x0.

        Uses the deterministic DDIM chain x_T -> x_{T-1} -> ... -> x_0 and
        accumulates log-probability through the Jacobian at each step:

            log p(x_0) = log p(x_T) - sum_t log |det (dx_{t-1}/dx_t)|

        The per-step Jacobian is:
            J_t = A_t * I + B_t * (d v_hat / d x_t)
            A_t = alpha_{t-1} * alpha_t + sigma_{t-1} * sigma_t
            B_t = sigma_{t-1} * alpha_t - alpha_{t-1} * sigma_t

        Caller should run only on events with two valid neutrinos (full 6D);
        masking logic for singular Jacobian is not applied.

        The model Jacobian (d v_hat / d x_t) is computed via reverse-mode AD,
        one backward pass per output dimension (6 passes for d=6 neutrinos).

        Args:
            data_shape: Shape of the data to generate (e.g. [B, 2, 3])
            pred_fn: Model prediction function (takes noise_x, time); outputs velocity
            normalize_fn: Optional denormalization function
            num_steps: Number of DDIM steps (default 100)
            eta: Stochasticity parameter (unused in deterministic DDIM)
            noise_mask: Optional mask for noise, shape [B, N, 1]
            use_tqdm: Whether to show progress bar
            process_name: Name for progress bar
            remove_padding: Whether to remove padding in denormalization

        Returns:
            x: Generated sample, shape data_shape (denormalized if normalize_fn)
            log_prob: Log-probability in model (normalized) space, shape [B]
        """
        batch_size = data_shape[0]
        flat_d = 1
        for s in data_shape[1:]:
            flat_d *= s
        const_shape = (batch_size, *([1] * (len(data_shape) - 1)))

        x = self.prior_sde(data_shape)
        if noise_mask is not None:
            x = x * noise_mask

        # log p(x_T) = -d/2 * log(2*pi) - 0.5 * ||x_T||^2
        log_2pi = torch.log(torch.tensor(2.0 * torch.pi, device=self.device))
        log_prob = -0.5 * flat_d * log_2pi - 0.5 * x.flatten(1).pow(2).sum(dim=1)  # [B]

        iterable = range(num_steps, 0, -1)
        if use_tqdm:
            iterable = tqdm(iterable, desc=process_name, total=num_steps)

        for time_step in iterable:
            t = torch.ones((batch_size,), device=self.device).float() * time_step / num_steps
            logsnr, alpha, sigma = get_logsnr_alpha_sigma(t, shape=const_shape)

            t_prev = torch.ones((batch_size,), device=self.device).float() * (time_step - 1) / num_steps
            logsnr_, alpha_, sigma_ = get_logsnr_alpha_sigma(t_prev, shape=const_shape)

            # --- Forward pass WITH gradients for Jacobian ---
            # inference_mode(False) is required because Lightning's predict_step
            # runs under inference_mode, which enable_grad() cannot override.
            with torch.inference_mode(mode=False):
                x_in = x.clone().detach().requires_grad_(True)
                output = pred_fn(noise_x=x_in, time=t)

                out_flat = output.flatten(1)  # [B, d]

                # Compute d(output)/d(x_t) row by row via reverse-mode AD: [B, d, d]
                # Differentiate w.r.t. x_in (the leaf), then flatten the gradient.
                d_out_dx = torch.zeros(batch_size, flat_d, flat_d, device=self.device)
                for j in range(flat_d):
                    e_j = torch.zeros_like(out_flat)
                    e_j[:, j] = 1.0
                    grad_j = torch.autograd.grad(
                        out_flat, x_in,
                        grad_outputs=e_j,
                        retain_graph=(j < flat_d - 1),
                        create_graph=False,
                    )[0]  # [B, 2, 3]
                    d_out_dx[:, j, :] = grad_j.flatten(1)  # -> [B, d]

            # --- Compute A_t, B_t (velocity prediction) ---
            a_t = alpha.flatten(1)[:, 0]
            s_t = sigma.flatten(1)[:, 0]
            a_prev = alpha_.flatten(1)[:, 0]
            s_prev = sigma_.flatten(1)[:, 0]
            A_t = a_prev * a_t + s_prev * s_t
            B_t = s_prev * a_t - a_prev * s_t

            # J = A_t * I + B_t * d_out_dx,  shape [B, d, d]
            eye = torch.eye(flat_d, device=self.device)
            J_full = A_t[:, None, None] * eye + B_t[:, None, None] * d_out_dx

            _, log_abs_det = torch.linalg.slogdet(J_full)  # [B]
            log_prob = log_prob - log_abs_det

            # --- Standard DDIM update (velocity -> eps -> pred_x0 -> x_{t-1}) ---
            with torch.no_grad():
                v = output.detach()
                eps = v * alpha + x * sigma
                pred_x0_val = (x - sigma * eps) / alpha
                x = alpha_ * pred_x0_val + sigma_ * eps
                if noise_mask is not None:
                    x = x * noise_mask

        if normalize_fn is not None:
            # --- Denormalization Jacobian correction ---
            # The denormalization is element-wise so its Jacobian is diagonal.
            # For features in inv_cdf_index:  dx/dz = phi(z) * 2*sqrt(3) * std
            # For other features:             dx/dz = std
            # log|det| = sum over all elements of log|dx_i/dz_i|
            if remove_padding and normalize_fn.padding > 0:
                current_std = normalize_fn.std[:-normalize_fn.padding]
            else:
                current_std = normalize_fn.std

            log_denorm_det = torch.log(current_std).sum() * x.shape[1]  # constant, broadcast over batch

            if len(normalize_fn.inv_cdf_index) > 0:
                z_cdf = x[..., normalize_fn.inv_cdf_index]  # [B, 2, len(inv_cdf)]
                log_phi = -0.5 * z_cdf.pow(2) - 0.5 * math.log(2 * math.pi)
                log_denorm_det = log_denorm_det + log_phi.sum(dim=(1, 2))  # [B]
                log_denorm_det = log_denorm_det + math.log(2 * math.sqrt(3)) * z_cdf.shape[1] * z_cdf.shape[2]

            log_prob = log_prob - log_denorm_det

            x = normalize_fn.denormalize(x, noise_mask, remove_padding=remove_padding)
        return x, log_prob
