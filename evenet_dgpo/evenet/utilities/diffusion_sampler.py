import torch
from torch import Tensor
from typing import Callable, Optional
from contextlib import nullcontext
from tqdm import tqdm

from evenet.utilities.debug_tool import time_decorator


def logsnr_schedule_cosine(time: Tensor, logsnr_min: float = -20., logsnr_max: float = 20.) -> Tensor:
    dtype = torch.float64 if time.dtype == torch.float64 else torch.float32
    time = time.to(dtype=dtype)
    logsnr_min = torch.tensor([logsnr_min], device=time.device, dtype=dtype)
    logsnr_max = torch.tensor([logsnr_max], device=time.device, dtype=dtype)
    b = torch.atan(torch.exp(-0.5 * logsnr_max)).to(time.device)
    a = (torch.atan(torch.exp(-0.5 * logsnr_min)) - b).to(time.device)
    return -2.0 * torch.log(torch.tan(a * time + b))


def get_logsnr_alpha_sigma(time: Tensor, shape=None):
    logsnr = logsnr_schedule_cosine(time)
    alpha = torch.sqrt(torch.sigmoid(logsnr))
    sigma = torch.sqrt(torch.sigmoid(-logsnr))

    if shape is not None:
        logsnr = logsnr.view(shape)
        alpha = alpha.view(shape)
        sigma = sigma.view(shape)

    return logsnr, alpha, sigma


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
    def __init__(self, device, *, x0_mode: str = "legacy", dtype=torch.float32):
        # Opt-in for matched sampling diagnostics; existing callers retain the
        # original arithmetic until the coverage ablation has been evaluated.
        if x0_mode not in ("legacy", "stable_v"):
            raise ValueError(f"Unknown DDIM x0_mode: {x0_mode}")
        self.device = device
        self.dtype = dtype
        self.x0_mode = x0_mode

    def prior_sde(self, dimensions) -> Tensor:
        return torch.randn(dimensions, dtype=self.dtype, device=self.device)

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
            differentiable: bool = False,
            checkpoint_steps: bool = False,
    ) -> Tensor:
        """
        time: time tensor (B,)

        Historical ``eta`` scales predicted epsilon; it is NOT the standard
        stochastic-DDIM eta. No per-step noise is added. Keep eta=1 for the
        deterministic update used by the matched coverage experiment.
        """
        batch_size = data_shape[0]
        const_shape = (batch_size, *([1] * (len(data_shape) - 1)))
        x = self.prior_sde(data_shape)
        if noise_mask is not None:
            x = x * noise_mask

        iterable = range(num_steps, 0, -1)
        if use_tqdm:
            iterable = tqdm(iterable, desc=process_name, total=num_steps)

        for time_step in iterable:
            t = torch.ones((batch_size,), device=self.device, dtype=x.dtype) * time_step / num_steps
            logsnr, alpha, sigma = get_logsnr_alpha_sigma(t, shape=const_shape)

            t_prev = torch.ones((batch_size,), device=self.device, dtype=x.dtype) * (time_step - 1) / num_steps
            logsnr_, alpha_, sigma_ = get_logsnr_alpha_sigma(t_prev, shape=const_shape)

            with nullcontext() if differentiable else torch.no_grad():

                # Compute the predicted epsilon using the model
                if differentiable and checkpoint_steps:
                    from torch.utils.checkpoint import checkpoint
                    v = checkpoint(pred_fn, noise_x=x, time=t, use_reentrant=False)
                else:
                    v = pred_fn(noise_x=x, time=t)

                eps = v * alpha + x * sigma

            # Update x using DDIM deterministic update rule
            if self.x0_mode == "stable_v":
                # VP rotation inverse: avoids cancellation/division by tiny
                # alpha at the high-noise endpoint for v-prediction models.
                pred_x0 = alpha * x - sigma * v
            else:
                pred_x0 = (x - sigma * eps) / alpha
            x = alpha_ * pred_x0 + sigma_ * (eta * eps)
            if noise_mask is not None:
                x = x * noise_mask

        if normalize_fn is not None:
            denormalize = normalize_fn.denormalize_grad if differentiable else normalize_fn.denormalize
            x = denormalize(x, noise_mask, remove_padding=remove_padding)
        return x
