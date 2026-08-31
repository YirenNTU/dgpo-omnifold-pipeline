"""Shared diffusion sampling interface for supervised, OmniFold, and DGPO stages."""

from __future__ import annotations

from functools import partial
from typing import Any

import torch
from torch import Tensor

from evenet.utilities.diffusion_sampler import DDIMSampler
from RL.DGPO_neutrino.dgpo_utils import repeat_batch_for_candidates


@torch.no_grad()
def generate_neutrino_candidates(
    model: torch.nn.Module,
    batch: dict[str, Any],
    sampler: DDIMSampler,
    *,
    K: int,
    num_ddim_steps: int,
    device: torch.device,
    parallel_chains: int = 1,
    tqdm_k_chains: bool = False,
    use_tqdm_ddim: bool = False,
    chain_progress_desc: str = "diffusion DDIM chains",
    expanded_batch: dict[str, Any] | None = None,
) -> Tensor:
    """Return every independent diffusion draw as ``(K, B, N, F)``.

    This function deliberately does not rank or select candidates. ``K=1`` is
    the unbiased population draw used to fit the current OmniFold classifier;
    DGPO can call the same function with its configured group size. When all K
    chains run together, ``expanded_batch`` can reuse an existing K-fold batch.
    """

    if int(K) < 1:
        raise ValueError(f"K must be positive, got {K}")
    if int(num_ddim_steps) < 1:
        raise ValueError(f"num_ddim_steps must be positive, got {num_ddim_steps}")
    if "x_invisible" not in batch:
        raise KeyError("batch missing x_invisible for DDIM data_shape")
    if "x_invisible_mask" not in batch:
        raise KeyError("batch missing x_invisible_mask for DDIM noise mask")

    batch_size, num_slots = batch["x_invisible"].shape[:2]
    invisible_dim = int(
        getattr(model, "invisible_input_dim", batch["x_invisible"].shape[-1])
    )
    parallel_chains = max(1, min(int(parallel_chains), int(K)))
    if expanded_batch is not None:
        if parallel_chains != int(K):
            raise ValueError(
                "expanded_batch can only be reused when all K chains run in one group"
            )
        expanded_x = expanded_batch.get("x")
        if (
            not isinstance(expanded_x, Tensor)
            or int(expanded_x.shape[0]) != int(K) * batch_size
        ):
            raise ValueError(
                "expanded_batch['x'] must have K*B rows; "
                f"expected {int(K) * batch_size}"
            )
    noise_mask = batch["x_invisible_mask"].unsqueeze(-1)

    def _sample_group(
        batch_group: dict[str, Any],
        noise_mask_group: Tensor,
        chain_count: int,
    ) -> Tensor:
        data_shape_group = (
            chain_count * batch_size,
            num_slots,
            invisible_dim,
        )
        pred_partial = partial(
            model.predict_diffusion_vector,
            mode="neutrino",
            cond_x=batch_group,
            noise_mask=noise_mask_group,
        )
        generated = sampler.sample(
            data_shape=data_shape_group,
            pred_fn=pred_partial,
            num_steps=int(num_ddim_steps),
            normalize_fn=model.invisible_normalizer,
            remove_padding=True,
            noise_mask=noise_mask_group,
            use_tqdm=use_tqdm_ddim,
            process_name=f"{chain_progress_desc} steps",
        )
        return generated.reshape(
            chain_count,
            batch_size,
            num_slots,
            generated.shape[-1],
        )

    # Common DGPO path: all K candidates fit in one K*B forward. Return the
    # reshaped sampler output directly, avoiding the group loop and a full-size
    # unbind -> list -> stack copy.
    if parallel_chains == int(K):
        if int(K) == 1:
            batch_group = batch
            noise_mask_group = noise_mask
        else:
            batch_group = (
                expanded_batch
                if expanded_batch is not None
                else repeat_batch_for_candidates(batch, int(K))
            )
            noise_mask_group = batch_group["x_invisible_mask"].unsqueeze(-1)
        return _sample_group(batch_group, noise_mask_group, int(K))

    candidate_groups: list[Tensor] = []
    k_iter: Any = range(0, int(K), parallel_chains)
    if tqdm_k_chains:
        try:
            from tqdm.auto import tqdm

            k_iter = tqdm(
                k_iter,
                desc=chain_progress_desc,
                leave=False,
                unit="group",
            )
        except ImportError:
            pass

    for chain_start in k_iter:
        chain_count = min(parallel_chains, int(K) - chain_start)
        if chain_count == 1:
            batch_group = batch
            noise_mask_group = noise_mask
        else:
            batch_group = repeat_batch_for_candidates(batch, chain_count)
            noise_mask_group = batch_group["x_invisible_mask"].unsqueeze(-1)
        candidate_groups.append(
            _sample_group(
                batch_group,
                noise_mask_group,
                chain_count,
            )
        )
    return torch.cat(candidate_groups, dim=0)


__all__ = ["generate_neutrino_candidates"]
