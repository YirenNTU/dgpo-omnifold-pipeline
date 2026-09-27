"""Visible-only Lorentz features. No truth labels, tau energy, or candidate boost.

Four-vectors use (E, px, py, pz), with metric (+---) and energies in GeV.
The rest frame is defined ONLY by visible_a + visible_b. The +/-z lab beam
directions are null reference vectors, not a hypothesis about a tau energy.
"""
from __future__ import annotations

import torch
from torch import Tensor


# v3 keeps eight compact channels; v1/v2 rest classifiers are incompatible.
REST_FRAME_KEY = "visible_pair_rest_v3"
REST_FRAME_FEATURE_NAMES = (
    "log1p_pair_mass_gev", "beta_z", "beta_transverse",
    "asinh_mass2_a_over_pair_mass2", "asinh_mass2_b_over_pair_mass2",
    "cos_a_cs_z", "cos_a_cs_x", "cos_a_cs_y",
)
REST_FRAME_DIM = len(REST_FRAME_FEATURE_NAMES)
VISIBLE_P4_KEYS = tuple(
    f"lead_{leg}_visible_{component}"
    for leg in ("a", "b") for component in ("E", "px", "py", "pz")
)


def mass_squared(p4: Tensor) -> Tensor:
    return p4[..., 0].square() - p4[..., 1:].square().sum(-1)


def boost_to_rest(p4: Tensor, frame: Tensor, *, validate: bool = True) -> Tensor:
    """Rotation-free boost INTO a future-timelike frame, with broadcasting.

    The spatial coefficient uses 1/[m(E_frame+m)], avoiding a beta**2
    denominator and cancellation in gamma-1 at a vanishing boost. Arithmetic
    is float64 even under autocast; output retains the input floating dtype.
    This is the sign convention of vector.boostCM_of_p4(), not boost_p4().
    """
    if p4.shape[-1] != 4 or frame.shape[-1] != 4:
        raise ValueError("boost_to_rest expects (...,4) E,px,py,pz tensors")
    if not p4.is_floating_point() or not frame.is_floating_point():
        raise TypeError("Lorentz vectors must be floating point")
    q, f = p4.double(), frame.double()
    m2 = mass_squared(f)
    if validate and not bool((torch.isfinite(q).all() & torch.isfinite(f).all()
                              & (f[..., 0] > 0).all() & (m2 > 0).all()).item()):
        raise ValueError("boost requires finite vectors and a future-timelike frame")
    mass = m2.sqrt()
    dot = (f[..., 1:] * q[..., 1:]).sum(-1)
    energy = (f[..., 0] * q[..., 0] - dot) / mass
    coefficient = dot / (mass * (f[..., 0] + mass)) - q[..., 0] / mass
    spatial = q[..., 1:] + coefficient[..., None] * f[..., 1:]
    return torch.cat((energy[..., None], spatial), -1).to(p4.dtype)


def _unit(vector: Tensor, eps: float = 1e-10) -> tuple[Tensor, Tensor]:
    squared = vector.square().sum(-1)
    valid = squared > eps * eps
    norm = squared.clamp_min(eps * eps).sqrt()
    unit = torch.where(valid[..., None], vector / norm[..., None], torch.zeros_like(vector))
    return unit, valid


def visible_pair_rest_features(visible_a: Tensor, visible_b: Tensor) -> Tensor:
    """Eight visible-only features in a versioned, checkpointed channel order.

    Remove log-gamma, rest energy/momentum ratios and beam-angle cosines:
    on valid nondegenerate events they follow from these retained channels.
    No feature selection is learned from truth labels or validation data.

    The Collins-Soper-style z axis bisects the boosted +z and reversed -z
    beam directions; x bisects the two forward beam directions; y=z cross x.
    The leg-a ordering is the dataset's ordering, NOT an inferred tau charge.
    At zero transverse recoil x/y are undefined: set those channels to zero,
    never choose an arbitrary lab azimuth. Validity checks are internal only;
    no validity/missingness flags are passed to the classifier.

    Non-finite input is an error. Finite null/spacelike/ill-conditioned total
    frames get neutral features, without dropping the event
    or applying a label-dependent cut. No rest-frame tau opening angle is used.
    """
    if visible_a.shape != visible_b.shape or visible_a.shape[-1] != 4:
        raise ValueError("visible legs must have the same (...,4) shape")
    if not visible_a.is_floating_point() or not visible_b.is_floating_point():
        raise TypeError("visible four-vectors must be floating point")
    a, b = visible_a.double(), visible_b.double()
    if not bool((torch.isfinite(a).all() & torch.isfinite(b).all()).item()):
        raise FloatingPointError("visible-pair rest features received NaN/Inf")
    total = a + b
    m2 = mass_squared(total)
    # Exclude numerically singular boosts; do not clip beta to invent a frame.
    valid = ((a[..., 0] > 0) & (b[..., 0] > 0)
             & (m2 > 1e-10 * total[..., 0].square().clamp_min(1.0)))
    rest = torch.zeros_like(total)
    rest[..., 0] = 1.0
    frame = torch.where(valid[..., None], total, rest)
    a_safe = torch.where(valid[..., None], a, rest * .5)
    b_safe = torch.where(valid[..., None], b, rest * .5)
    mass = mass_squared(frame).sqrt()
    a_star = boost_to_rest(a_safe, frame, validate=False)
    beam_plus = torch.zeros_like(frame)
    beam_plus[..., 0] = 1.0
    beam_plus[..., 3] = 1.0
    beam_minus = beam_plus.clone()
    beam_minus[..., 3] = -1.0
    plus, _ = _unit(boost_to_rest(beam_plus, frame, validate=False)[..., 1:])
    minus, _ = _unit(boost_to_rest(beam_minus, frame, validate=False)[..., 1:])
    z_axis, _ = _unit(plus - minus)
    x_axis, _ = _unit(plus + minus)
    y_axis = torch.linalg.cross(z_axis, x_axis, dim=-1)
    direction, _ = _unit(a_star[..., 1:] / mass[..., None])
    beta = frame[..., 1:] / frame[..., :1]

    def dot(left: Tensor, right: Tensor) -> Tensor:
        return (left * right).sum(-1).clamp(-1.0, 1.0)

    channels = (
        torch.log1p(mass),
        beta[..., 2], torch.linalg.vector_norm(beta[..., :2], dim=-1),
        torch.asinh(mass_squared(a_safe) / mass.square()),
        torch.asinh(mass_squared(b_safe) / mass.square()),
        dot(direction, z_axis), dot(direction, x_axis), dot(direction, y_axis),
    )
    features = torch.stack(channels, -1)
    features = torch.where(valid[..., None], features, torch.zeros_like(features))
    if not bool(torch.isfinite(features).all().item()):
        raise FloatingPointError("non-finite visible-pair rest features")
    return features.to(visible_a.dtype)
