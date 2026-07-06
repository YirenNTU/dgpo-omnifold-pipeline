import torch
from torch import Tensor
from typing import Optional


def _event_weight_for_broadcast(event_weight: Tensor, reference: Tensor) -> Tensor:
    """Reshape per-event weights from ``(B,)`` to broadcast over ``reference``."""
    if event_weight.dim() != 1 or event_weight.shape[0] != reference.shape[0]:
        raise ValueError(
            "event_weight must have shape (B,), got "
            f"{tuple(event_weight.shape)} for reference {tuple(reference.shape)}"
        )
    return event_weight.reshape((event_weight.shape[0],) + (1,) * (reference.dim() - 1))


def loss(
        predict: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
        feature_dim: Optional[int] = None,
        event_weight: Optional[Tensor] = None
):
    if event_weight is not None:
        event_weight = event_weight.to(device=predict.device, dtype=predict.dtype).reshape(-1)
        event_weight_b = _event_weight_for_broadcast(event_weight, predict)
        if mask is not None:
            mask = mask.float()
            den = torch.sum(mask * event_weight_b) * feature_dim
            if den == 0:
                return torch.tensor(0.0, device=predict.device, dtype=predict.dtype)
            return torch.sum(((predict - target) ** 2) * mask * event_weight_b) / den
        else:
            return torch.sum(((predict - target) ** 2) * event_weight_b) / torch.sum(event_weight)
    else:
        if mask is not None:
            den = torch.sum(mask.float()) * feature_dim
            if den == 0:
                return torch.tensor(0.0, device=predict.device, dtype=predict.dtype)
            return torch.sum(((predict - target) ** 2) * mask.float()) / den
        else:
            return torch.mean((predict - target) ** 2)
