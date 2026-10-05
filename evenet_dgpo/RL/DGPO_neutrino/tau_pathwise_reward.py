"""Differentiable mirror of the saved tau reward, including its frozen trunk.

Only observed conditioning may cross NumPy. Candidate-dependent computations
stay in Torch. Parameters/buffers are frozen; input derivatives remain enabled.
The historical fixed-energy reconstruction is a feature map, not unfolding.
"""
from __future__ import annotations

import torch
from torch import nn


def candidate_coordinates_grad(visible_a, visible_b, deltas, stats):
    directions, relatives = [], []
    for index, visible in enumerate((visible_a, visible_b)):
        visible = visible.to(device=deltas.device, dtype=torch.float64)
        delta = deltas[:, index].double()
        p = visible[:, 1:]
        if (torch.linalg.vector_norm(p, dim=1) == 0).any():
            raise ValueError('Undefined visible direction')
        theta_v = torch.atan2(torch.linalg.vector_norm(p[:, :2], dim=1), p[:, 2])
        phi_v = torch.atan2(p[:, 1], p[:, 0])
        theta, phi = theta_v + delta[:, 0], phi_v + delta[:, 1]
        # Match tau_from_deltas and the canonical angles after pole reflection.
        mag = ((91.2 / 2)**2 - 1.777**2)**.5
        q = mag * torch.stack((theta.sin()*phi.cos(), theta.sin()*phi.sin(), theta.cos()), 1)
        theta_q = torch.atan2(torch.linalg.vector_norm(q[:, :2], dim=1), q[:, 2])
        phi_q = torch.atan2(q[:, 1], q[:, 0])
        relatives.append(torch.stack((theta_q-theta_v, (phi_q-phi_v).sin(), (phi_q-phi_v).cos()), 1))
        directions.append(q / torch.linalg.vector_norm(q, dim=1, keepdim=True))
    relative = torch.cat(relatives, 1)
    mean = torch.as_tensor(stats['mean'], device=deltas.device, dtype=torch.float64)
    scale = torch.as_tensor(stats['scale'], device=deltas.device, dtype=torch.float64)
    a, b = directions
    # NumPy baseline casts tau_features to float32 before concatenation.
    features = torch.cat((a, b, (a[:, :, None]*b[:, None, :]).flatten(1)), 1).float()
    return torch.cat(((relative-mean)/scale, features.double()), 1).float()


def candidate_hidden_grad(policy, batch, candidate):
    mask = torch.ones((*candidate.shape[:2], 1), device=candidate.device)
    normalizer = policy.invisible_coordinate_normalizer(batch)
    pad = policy.invisible_normalizer.mean.numel() - candidate.shape[-1]
    if pad < 0:
        raise ValueError('Candidate width exceeds native normalizer')
    normalized = normalizer.forward_grad(nn.functional.pad(candidate, (0, pad)), mask=mask)
    captured = []
    handle = policy.TruthGeneration.generator.register_forward_pre_hook(
        lambda module, args: captured.append(args[0][:, -2:]))
    try:
        policy.predict_diffusion_vector(normalized, batch,
            torch.zeros(len(candidate), device=candidate.device), 'neutrino', noise_mask=mask)
    finally:
        handle.remove()
    if len(captured) != 1 or not torch.isfinite(captured[0]).all():
        raise ValueError('Expected one finite differentiable pre-velocity hidden tensor')
    return captured[0].flatten(1)


def compute_reward_grad(source, candidates, batch, *, prepared=None):
    from RL.DGPO_neutrino.conditional_tau_reward import (
        feature_precision,
    )
    from RL.DGPO_neutrino.rewards import apply_event_valid_to_rewards
    if candidates.ndim != 4 or candidates.shape[-2:] != (2, 2):
        raise ValueError('Expected K,B,2,2 angular candidates')
    if getattr(source, '_tau_diagnostic_ensemble', None) is not None:
        raise ValueError('Pathwise ablation requires the inherited single frozen reward')
    for module in (source.head, source.backbone):
        if module.training or any(p.requires_grad for p in module.parameters()):
            raise ValueError('Classifier head and trunk must be eval/frozen')
    prepared = source.prepare_inputs(batch) if prepared is None else prepared
    condition, visible = prepared['condition_tensor'], prepared['visible']
    trunk_batch = prepared['trunk_batch']
    scores = []
    with feature_precision():
        for delta in candidates:
            hidden = candidate_hidden_grad(source.backbone, trunk_batch, delta)
            coordinates = candidate_coordinates_grad(*visible, delta, source.bundle['head']['relative_preprocessing'])
            features = torch.cat((hidden, coordinates), 1).float()
            if features.shape[1] != source.bundle['head']['candidate_dim']:
                raise ValueError('Candidate feature width changed')
            scores.append(source.head(condition, features))
    result = apply_event_valid_to_rewards(torch.stack(scores), batch)
    if not torch.isfinite(result).all():
        raise ValueError('Nonfinite differentiable tau reward')
    return result
