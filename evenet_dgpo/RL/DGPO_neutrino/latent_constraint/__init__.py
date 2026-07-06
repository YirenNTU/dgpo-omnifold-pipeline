"""Latent-space constraint model for DGPO neutrino reconstruction.

A compact, independently trainable autoencoder that learns an event/neutrino
latent space. The single supported model is the **object-token bottleneck AE**
(:class:`ObjectTokenBottleneckAutoencoder`): inputs are the frozen EveNet event
CLS token plus all per-object ObjectEncoder tokens together with the neutrino
kinematics; the target is the original pretrain-model event token and the
neutrino kinematics. After training, its checkpoint is loaded inside DGPO as a
frozen constraint encoder: truth and predicted neutrino configurations are
mapped into the same latent space and compared with sliced Wasserstein distance
to produce the CPO constraint source.

Public API
----------
- :class:`ObjectTokenBottleneckAutoencoder` — the model (``encode_latent`` / ``decode``).
- :func:`load_checkpoint` / :func:`save_checkpoint` — checkpoint I/O for DGPO use.
- :func:`sliced_wasserstein_distance` — differentiable SWD utility.
- :func:`random_projections` — projection-direction sampler (deterministic option).
- :func:`load_normalizers_from_pt` — normalizer triple from ``normalization.pt``.
"""

from RL.DGPO_neutrino.latent_constraint.normalizers import load_normalizers_from_pt
from RL.DGPO_neutrino.latent_constraint.object_token_ae import (
    ObjectTokenBottleneckAutoencoder,
    load_checkpoint,
    parse_lc_resume_from_checkpoint,
    save_checkpoint,
)
from RL.DGPO_neutrino.latent_constraint.sliced_wasserstein import (
    random_projections,
    sliced_wasserstein_distance,
)

__all__ = [
    "ObjectTokenBottleneckAutoencoder",
    "load_checkpoint",
    "load_normalizers_from_pt",
    "parse_lc_resume_from_checkpoint",
    "save_checkpoint",
    "random_projections",
    "sliced_wasserstein_distance",
]
