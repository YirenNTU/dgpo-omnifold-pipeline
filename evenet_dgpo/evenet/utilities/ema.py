import copy
import torch
import torch.nn as nn


def resolve_ema_update_every_n_steps(config) -> int:
    """Return the configured EMA update interval.

    ``update_every_n_steps`` is the canonical YAML key.  ``update_step`` is
    retained as a fallback for checkpoints or configs written before the key
    was standardized.
    """

    raw_interval = config.get(
        "update_every_n_steps",
        config.get("update_step", 1),
    )
    if isinstance(raw_interval, bool):
        raise ValueError("EMA update_every_n_steps must be a positive integer")
    try:
        interval = int(raw_interval)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "EMA update_every_n_steps must be a positive integer"
        ) from exc
    if interval < 1 or (
        isinstance(raw_interval, float) and not raw_interval.is_integer()
    ):
        raise ValueError("EMA update_every_n_steps must be a positive integer")
    return interval


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.current_epoch = 0

        self.shadow = {}
        self.model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module, decay_: float = None):
        model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        decay = self.decay if decay_ is None else decay_

        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(decay).add_(param.data, alpha=1.0 - decay)

    def copy_to(self, model: nn.Module):
        model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict, device=None):
        self.shadow = {k: v.clone().to(device) for k, v in state_dict.items()}
