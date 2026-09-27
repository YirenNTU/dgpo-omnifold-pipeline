"""Opt-in controls for matched supervised Fourier integration experiments."""
from contextlib import contextmanager

import torch


def load_no_fourier_weights(model, checkpoint):
    """Load raw shared weights exactly, allowing only new angular parameters."""
    source = {k.removeprefix("model."): v for k, v in checkpoint["state_dict"].items()}
    prefix = "PET.angular_conditioning."
    if any(k.startswith(prefix) for k in source):
        raise ValueError("The integration experiment requires a no-Fourier source checkpoint")
    target = model.state_dict()
    missing = [k for k in target if k not in source and not k.startswith(prefix)]
    mismatched = [k for k in target if k in source and target[k].shape != source[k].shape]
    # Disabled heads and FAMO (added after loading) may remain in the source.
    unused_roots = {"famo", "Classification", "Regression", "Assignment", "Segmentation",
                    "GlobalGeneration", "ReconGeneration"}
    unexpected = [k for k in source if k not in target and k.split(".")[0] not in unused_roots]
    if missing or mismatched or unexpected:
        raise ValueError(f"Incompatible shared weights: missing={missing}, "
                         f"shape_mismatch={mismatched}, unexpected={unexpected}")
    model.load_state_dict({k: source[k] for k in target if k in source}, strict=False)


def optimizer_parameters(model, module_paths, all_module_paths):
    """Give explicitly configured child modules ownership of their parameters."""
    params, seen = [], set()
    for path in module_paths:
        excluded = {
            id(p) for child in all_module_paths if child.startswith(path + ".")
            for p in model.get_submodule(child).parameters()
        }
        for p in model.get_submodule(path).parameters():
            if id(p) not in excluded and id(p) not in seen:
                params.append(p)
                seen.add(id(p))
    return params


def load_conditioning_ablation_weights(model, checkpoint):
    """Exact shared raw load for PET-only/global-FiLM/token-FiLM continuation.

    Dropping global FiLM is allowed only when all saved modulation outputs are
    identically zero. Adding the new token readout is also a zero-output change.
    Any other missing, unexpected, or mismatched shared parameter is an error.
    """
    source = {k.removeprefix("model."): v for k, v in checkpoint["state_dict"].items()}
    target = model.state_dict()
    prefix = "TruthGeneration.visible_conditioning."
    new_prefix = prefix + "token_readout."
    removing_global = any(k.startswith(prefix) for k in source) and not any(k.startswith(prefix) for k in target)
    if removing_global:
        gates = [v for k, v in source.items() if k.startswith(prefix + "modulations.")]
        if not gates or any(torch.count_nonzero(v).item() for v in gates):
            raise ValueError("PET-only arm can remove global FiLM only from a zero-output step-0 source")
        if any(k.startswith(new_prefix) for k in source):
            raise ValueError("common supervised source must precede token-readout training")
    unused_roots = {"famo", "Classification", "Regression", "Assignment", "Segmentation", "GlobalGeneration", "ReconGeneration"}
    missing = [k for k in target if k not in source and not k.startswith(new_prefix)]
    mismatch = [k for k in target if k in source and target[k].shape != source[k].shape]
    unexpected = [k for k in source if k not in target and k.split(".")[0] not in unused_roots
                  and not (removing_global and k.startswith(prefix))]
    if missing or mismatch or unexpected:
        raise ValueError(f"Incompatible conditioning ablation source: missing={missing}, shape={mismatch}, unexpected={unexpected}")
    model.load_state_dict({k: source[k] for k in target if k in source}, strict=False)


@contextmanager
def paired_diffusion_rng(seed, *, training, epoch, batch_idx, rank, device):
    """Pair rank/batch draws; validation draws stay fixed across epochs.

    Requires the same ordered batches and worker count; changing data sharding
    does not preserve event-level noise pairing.
    """
    if seed is None:
        yield
        return
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    draw_seed = int(seed) + 1_000_003 * int(rank) + int(batch_idx)
    draw_seed += (10_000_019 * (int(epoch) + 1)) if training else 1_000_000_007
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(draw_seed)
        if devices:
            with torch.cuda.device(devices[0]):
                torch.cuda.manual_seed(draw_seed)
        yield
