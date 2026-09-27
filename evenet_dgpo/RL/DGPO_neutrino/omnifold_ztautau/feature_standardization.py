"""Fixed, training-population-only Fourier output scaling for a cold-fit ablation."""
import torch
from torch import nn


class FixedFeatureStandardizer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.register_buffer('mean', torch.zeros(width))
        self.register_buffer('scale', torch.ones(width))

    def forward(self, x):
        return (x - self.mean) / self.scale

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Old banks predate this optional identity transform. Partial state is an error.
        if prefix + 'mean' not in state_dict and prefix + 'scale' not in state_dict:
            state_dict[prefix + 'mean'] = torch.zeros_like(self.mean)
            state_dict[prefix + 'scale'] = torch.ones_like(self.scale)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


@torch.no_grad()
def fit_output_standardizer(model, populations, rank=0, world=1, cap=8192):
    """Balanced weighted moments on first <=8192 fit rows/class, sharded over ranks.

    Populations are the replicated, flattened training pools used by ratio_fit.
    No validation input is accepted. Eval-mode statistics are frozen afterwards.
    """
    from .stability_diagnostic import diagnostic_modules, diagnostic_buffers, rng_state, restore_rng
    bank = getattr(model, 'bank', None)
    if (bank is None or not hasattr(bank, 'topology_standardizer')
            or bank.topology_conditioning or bank.topology_direct_logit):
        raise ValueError('Output standardization requires Fourier late fusion without direct/conditioning')
    device = next(model.parameters()).device
    modules = diagnostic_modules(model)
    modes = {k: m.training for k, m in modules.items()}
    buffers = {k: b.clone() for k, b in diagnostic_buffers(model).items()}
    rng = rng_state(device)
    width = bank.topology_standardizer.mean.numel()
    sums = torch.zeros(2, 2 * width + 1, device=device, dtype=torch.float64)
    error = torch.zeros((), device=device)
    captured = []
    handle = bank.topology_encoder.register_forward_hook(lambda m, a, out: captured.append(out.detach()))
    try:
        model.eval()
        for label, (c, z, weights) in enumerate(populations):
            indices = torch.arange(rank, min(len(z), cap), world, device=device)
            for index in indices.split(256):
                if not len(index):
                    continue
                captured.clear()
                model(c[index], z[index])
                if len(captured) != 1:
                    raise ValueError('Expected exactly one Fourier encoder call')
                values, w = captured[0].double(), weights[index].double()
                if not torch.isfinite(values).all() or not torch.isfinite(w).all() or (w < 0).any():
                    raise ValueError('Invalid fit features/weights for standardization')
                sums[label, :width] += (values * w[:, None]).sum(0)
                sums[label, width:2 * width] += (values.square() * w[:, None]).sum(0)
                sums[label, -1] += w.sum()
    except Exception:
        error.fill_(1)
    finally:
        handle.remove()
        for k, module in modules.items():
            module.training = modes[k]
        for k, buffer in diagnostic_buffers(model).items():
            buffer.copy_(buffers[k])
        restore_rng(rng, device)
    if world > 1:
        torch.distributed.all_reduce(error, op=torch.distributed.ReduceOp.MAX)
        torch.distributed.all_reduce(sums)
    if error.item() or (sums[:, -1] <= 0).any():
        raise ValueError('Distributed training-only Fourier calibration failed')
    moments = sums[:, :-1] / sums[:, -1:]
    mean = moments[:, :width].mean(0)
    variance = (moments[:, width:].mean(0) - mean.square()).clamp_min(0)
    scale = variance.sqrt().clamp_min(1e-6)
    bank.topology_standardizer.mean.copy_(mean)
    bank.topology_standardizer.scale.copy_(scale)
    return dict(scale_min=float(scale.min()), scale_max=float(scale.max()),
                floored_fraction=float((variance.sqrt() < 1e-6).double().mean()),
                fit_rows_per_class=min(cap, len(populations[0][1]), len(populations[1][1])))
