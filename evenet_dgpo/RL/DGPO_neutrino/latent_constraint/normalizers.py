"""Normalizer construction from EveNet ``normalization.pt`` files.

Home of :func:`load_normalizers_from_pt` (moved here from the removed
``RL/CPO/discriminator_model.py`` when the classifier constraint was dropped).
Builds the visible / conditions / invisible :class:`Normalizer` triple the
latent-constraint encoder shares with the EveNet policy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from evenet.network.body.normalizer import Normalizer


def load_normalizers_from_pt(
    normalization_path: str | Path,
    *,
    device: torch.device | None = None,
    cartesian: bool = False,
) -> tuple[Normalizer, Normalizer, Normalizer]:
    """
    Build visible, conditions, and invisible normalizers from a ``normalization.pt`` file.

    Uses the same keys as EveNet preprocessing / ``normalization.pt``:
    ``input_mean/std['Source']``, ``input_mean/std['Conditions']``, and for neutrino
    kinematics ``invisible_mean/std['Source']`` (``log_pt, eta, phi``) or, when
    ``cartesian=True``, ``invisible_cartesian_mean/std['Source']`` (``px, py, pz``).
    The cartesian keys mirror what the main EveNet model reads when
    ``TruthGeneration.cartesian`` is set (see ``evenet_model.py``).
    """
    path = Path(normalization_path).expanduser().resolve()
    norm_dict: dict[str, Any] = torch.load(str(path), map_location="cpu", weights_only=False)
    dev = device or torch.device("cpu")

    seq_mean = norm_dict["input_mean"]["Source"].to(dev)
    seq_std = norm_dict["input_std"]["Source"].to(dev)
    n_seq = int(seq_mean.numel())
    sequential = Normalizer(
        mean=seq_mean,
        std=seq_std,
        norm_mask=torch.ones(n_seq, dtype=torch.bool, device=dev),
        inv_cdf_index=[],
        padding_size=0,
    )

    cond_mean = norm_dict["input_mean"]["Conditions"].to(dev)
    cond_std = norm_dict["input_std"]["Conditions"].to(dev)
    n_cond = int(cond_mean.numel())
    global_norm = Normalizer(
        mean=cond_mean,
        std=cond_std,
        norm_mask=torch.ones(n_cond, dtype=torch.bool, device=dev),
        inv_cdf_index=[],
        padding_size=0,
    )

    inv_mean_key = "invisible_cartesian_mean" if cartesian else "invisible_mean"
    inv_std_key = "invisible_cartesian_std" if cartesian else "invisible_std"
    if cartesian and (inv_mean_key not in norm_dict or inv_std_key not in norm_dict):
        raise ValueError(
            "cartesian=True requires invisible_cartesian_mean and "
            "invisible_cartesian_std in normalization.pt. Re-run preprocessing to "
            "generate them (same requirement as the main EveNet cartesian path)."
        )
    inv_mean = norm_dict[inv_mean_key]["Source"].to(dev)
    inv_std = norm_dict[inv_std_key]["Source"].to(dev)
    n_inv = int(inv_mean.numel())
    invisible = Normalizer(
        mean=inv_mean,
        std=inv_std,
        norm_mask=torch.ones(n_inv, dtype=torch.bool, device=dev),
        inv_cdf_index=[],
        padding_size=0,
    )

    return sequential, global_norm, invisible
