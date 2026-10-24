"""Opt-in truth-assisted, within-event General/oracle mixture for DGPO training.

Physical offsets only. Truth chooses ONE complete two-leg candidate. The General
network never receives truth; validation uses unweighted generator samples.
"""
import math
import torch

LEGACY_GEOMETRY = 'sum_two_leg_squared_geodesic'
JOINT_GEOMETRY = 'normalized_two_leg_axis_signed_polar_v1'
POLAR_GEOMETRY = 'normalized_signed_polar_sum_v1'
MOMENTUM_GEOMETRY = 'normalized_fixed_energy_two_leg_momentum_v1'
RECO_MC_GEOMETRY = 'normalized_reconstructed_mc_tau_p4_v1'
MC_TAU_COLUMNS = tuple(f'truth_tau_{leg}_{q}' for leg in 'ab' for q in ('E','px','py','pz'))


def geometry_options(config):
    geometry = config.get('geometry', LEGACY_GEOMETRY)
    scales = config.get('scales_mean_squared')
    energy, mass = config.get('fixed_energy_GeV'), config.get('tau_mass_GeV')
    if geometry in (MOMENTUM_GEOMETRY, RECO_MC_GEOMETRY):
        if (scales is not None or energy is None or mass is None
                or not math.isfinite(float(energy)) or not math.isfinite(float(mass))
                or not float(energy) > float(mass) >= 0):
            raise ValueError('Momentum oracle requires finite E > mass >= 0 and common p0 normalization')
        opts = dict(geometry=geometry, fixed_energy_GeV=float(energy), tau_mass_GeV=float(mass))
        if geometry == RECO_MC_GEOMETRY:
            if config.get('reconstruction') != 'raw_directions_no_post_calibration':
                raise ValueError('MC p4 oracle requires explicit uncalibrated two-leg directions')
            opts['reconstruction'] = config['reconstruction']
        return opts
    if energy is not None or mass is not None:
        raise ValueError('Fixed-energy parameters are only defined for the momentum oracle')
    if geometry in (JOINT_GEOMETRY, POLAR_GEOMETRY):
        count = 3 if geometry == JOINT_GEOMETRY else 1
        if (not isinstance(scales, (list, tuple)) or len(scales) != count
                or any(not math.isfinite(float(s)) or float(s) <= 0 for s in scales)):
            raise ValueError(f'{geometry} needs {count} fixed positive mean-squared scales')
    elif geometry != LEGACY_GEOMETRY or scales is not None:
        raise ValueError('Unknown oracle geometry or scales attached to legacy oracle')
    return dict(geometry=geometry, scales_mean_squared=scales)


def joint_contract(config):
    """Persist the preference independently of replaceable General heads."""
    opts = geometry_options(config)
    if opts['geometry'] not in (JOINT_GEOMETRY, POLAR_GEOMETRY, MOMENTUM_GEOMETRY, RECO_MC_GEOMETRY):
        return None
    alpha = .3 if opts['geometry'] == RECO_MC_GEOMETRY else .5 if opts['geometry'] == MOMENTUM_GEOMETRY else .1
    if (not config.get('enabled') or config.get('alpha') != alpha or config.get('adaptive')
            or config.get('reward', 'log_convex_weights') != 'log_convex_weights'):
        raise ValueError(f'This structured oracle experiment requires fixed oracle fraction {alpha}')
    return dict(**opts, alpha=alpha, reward='log_convex_weights', version=1)


def validate_resume_contract(config, checkpoint):
    expected = joint_contract(config)
    saved = (checkpoint.get('dgpo_omnifold_reward_stack') or {}).get('joint_oracle_contract')
    if (saved is not None or (expected is not None and checkpoint.get('dgpo_checkpoint_version'))) and saved != expected:
        raise ValueError('Joint oracle resume geometry/scales/mixture changed or contract missing')


def directions(offsets, visible):
    theta = torch.atan2(torch.linalg.vector_norm(visible[..., :2], dim=-1), visible[..., 2])
    phi = torch.atan2(visible[..., 1], visible[..., 0])
    theta, phi = theta + offsets[..., 0], phi + offsets[..., 1]
    return torch.stack((theta.sin()*phi.cos(), theta.sin()*phi.sin(), theta.cos()), -1)


def validate_mc_tau_p4(p4):
    """Actual MC E,px,py,pz, without fixing its energy or projecting its direction."""
    if p4.ndim != 3 or p4.shape[1:] != (2,4) or not torch.isfinite(p4).all():
        raise ValueError('Require finite actual MC tau p4 with shape B,2,4 (E,px,py,pz)')
    energy, momentum = p4[...,0], p4[...,1:]
    p2 = momentum.square().sum(-1)
    # Permit only floating-point roundoff, not genuinely spacelike MC inputs.
    if ((energy <= 0).any() or (p2 <= 0).any()
            or (energy.square()-p2 < -1e-5*energy.square()).any()):
        raise ValueError('Invalid actual MC tau energy/momentum; no truth substitution allowed')


def mc_tau_p4_from_batch(batch, device):
    missing = [key for key in MC_TAU_COLUMNS if key not in batch]
    if missing:
        raise ValueError('Actual MC tau p4 columns missing: '+', '.join(missing))
    return torch.stack([torch.stack([batch[f'truth_tau_{leg}_{q}'].reshape(-1)
        for q in ('E','px','py','pz')],-1) for leg in 'ab'],1).to(device=device,dtype=torch.float64)


@torch.no_grad()
def mix_reward(scores, candidates, truth, visible, alpha=0.1, *,
               geometry=LEGACY_GEOMETRY, scales_mean_squared=None,
               fixed_energy_GeV=None, tau_mass_GeV=None, reconstruction=None,
               truth_tau_p4=None):
    """Inputs K,B logits; K,B,2,2 candidates; B,2,2 truth; B,2,3 visible."""
    if not 0 <= alpha < 1:
        raise ValueError('Oracle mixture requires 0 <= alpha < 1')
    k, b = scores.shape
    if k < 2 or candidates.shape != (k,b,2,2) or truth.shape != (b,2,2) or visible.shape != (b,2,3):
        raise ValueError('Oracle requires two physical angular offsets per leg and K>=2')
    if any(not torch.isfinite(v).all() for v in (scores,candidates,truth,visible)):
        raise ValueError('Nonfinite oracle inputs; do not silently repair truth/candidates')
    if (torch.linalg.vector_norm(visible,dim=-1) <= 0).any():
        raise ValueError('Undefined visible direction')
    # Double precision keeps nearly collinear nearest-candidate decisions stable.
    u = directions(candidates.double(),visible.double())
    t = directions(truth.double(),visible.double()).unsqueeze(0).expand_as(u)
    angles = torch.atan2(torch.linalg.vector_norm(torch.linalg.cross(u,t),dim=-1),(u*t).sum(-1))
    geometry_options(dict(geometry=geometry, scales_mean_squared=scales_mean_squared,
                          fixed_energy_GeV=fixed_energy_GeV, tau_mass_GeV=tau_mass_GeV,
                          reconstruction=reconstruction))
    if geometry != RECO_MC_GEOMETRY and truth_tau_p4 is not None:
        raise ValueError('Actual MC p4 is only used by the explicitly selected reconstructed MC oracle')
    leg_distance = angles.square().sum(-1)
    distance = leg_distance
    terms = None
    if geometry == JOINT_GEOMETRY:
        axis = u[..., 0, :] - u[..., 1, :]
        target_axis = t[..., 0, :] - t[..., 1, :]
        norms = torch.linalg.vector_norm(axis, dim=-1, keepdim=True)
        target_norms = torch.linalg.vector_norm(target_axis, dim=-1, keepdim=True)
        if (norms <= 1e-12).any() or (target_norms <= 1e-12).any():
            raise ValueError('Undefined ordered tau-pair axis')
        axis, target_axis = axis/norms, target_axis/target_norms
        axis_angle = torch.atan2(torch.linalg.vector_norm(torch.linalg.cross(axis,target_axis),dim=-1),
                                 (axis*target_axis).sum(-1))
        # Visible theta cancels in the SIGNED polar-sum difference. No abs,
        # angular wrapping or forced back-to-back calibration in this term.
        polar_error = (candidates.double()[...,0] - truth.double()[None,...,0]).sum(-1)
        terms = torch.stack((leg_distance/2, axis_angle.square(), polar_error.square()), -1)
        scales = torch.as_tensor(scales_mean_squared, device=terms.device, dtype=terms.dtype)
        distance = (terms/scales).sum(-1)
    elif geometry == POLAR_GEOMETRY:
        # Preserve the SIGN of theta_a+theta_b-pi until comparing candidate
        # with truth. Visible theta cancels exactly; do not fold abs(s) first.
        # One shared frozen scale normalizes units without changing ranking.
        polar_error = (candidates.double()[...,0] - truth.double()[None,...,0]).sum(-1)
        terms = polar_error.square().unsqueeze(-1)
        distance = terms[...,0] / float(scales_mean_squared[0])
    elif geometry == MOMENTUM_GEOMETRY:
        # Same fixed energy for both candidate and truth anchor, equal legs and
        # Cartesian components. This is a chord metric, not energy prediction.
        # Keep the complete two-leg candidate; do not impose back-to-back.
        p0 = math.sqrt(float(fixed_energy_GeV)**2 - float(tau_mass_GeV)**2)
        distance = ((p0*u - p0*t).square().sum(-1) / p0**2).mean(-1)
    elif geometry == RECO_MC_GEOMETRY:
        if truth_tau_p4 is None:
            raise ValueError('Actual MC tau p4 required; never substitute fixed-energy truth directions')
        mc = truth_tau_p4.to(device=u.device,dtype=torch.float64)
        validate_mc_tau_p4(mc)
        if mc.shape[0] != b:
            raise ValueError('MC p4 batch size differs from candidates')
        mc_direction = mc[...,1:]/torch.linalg.vector_norm(mc[...,1:],dim=-1,keepdim=True)
        if (torch.linalg.vector_norm(mc_direction-t[0],dim=-1) > 1e-4).any():
            raise ValueError('Actual MC p4 disagrees with physical truth offsets/ordered visible legs')
        p0 = math.sqrt(float(fixed_energy_GeV)**2-float(tau_mass_GeV)**2)
        # Positive Euclidean distance in E,px,py,pz, NOT a Minkowski interval.
        # E is fixed for candidates only. Its error is constant across K; actual
        # MC momentum magnitudes still affect the relative importance of legs.
        # Equal legs/Cartesian axes and one common fixed p0 scale; no B2B step.
        spatial = (p0*u-mc[None,...,1:]).square().sum(-1)
        energy = (float(fixed_energy_GeV)-mc[None,...,0]).square()
        distance = ((spatial+energy)/p0**2).mean(-1)
    best = distance.argmin(0)
    log_general = scores.double().log_softmax(0)
    if alpha == 0:
        mixed = log_general
    else:
        oracle = torch.full_like(log_general,-torch.inf)
        oracle.scatter_(0,best[None],math.log(alpha))
        mixed = torch.logaddexp(log_general + math.log1p(-alpha),oracle)
    weights = mixed.exp()
    return mixed.to(scores.dtype), dict(weights=weights, general_weights=log_general.exp(),
        best=best, distance=distance, leg_distance=leg_distance, squared_terms=terms)


@torch.no_grad()
def training_reward(scores, candidates, batch, alpha=0.1, **geometry):
    # EveNet normalizes internally: x_invisible in the loader is already physical.
    truth = batch['x_invisible'].to(candidates)
    visible = torch.stack([torch.stack([batch[f'lead_{leg}_visible_{q}'].reshape(-1)
        for q in ('px','py','pz')],-1) for leg in 'ab'],1).to(candidates)
    # Masks/padding are applied by the native advantage/loss path. Require valid
    # tau legs; these are the filtered two-leg angular training targets.
    mask = batch['x_invisible_mask']
    if not torch.all(mask > 0):
        raise ValueError('Oracle mixture requires both tau legs to be present')
    mc = {}
    if geometry.get('geometry') == RECO_MC_GEOMETRY:
        mc['truth_tau_p4'] = mc_tau_p4_from_batch(batch,candidates.device)
    reward, info = mix_reward(scores,candidates,truth,visible,alpha,**geometry,**mc)
    w = info['weights']; chosen = info['best'][None]
    metrics = {'oracle/alpha': float(alpha),
        'oracle/mean_ess': float((1/w.square().sum(0)).mean()),
        'oracle/general_mass': float(info['general_weights'].gather(0,chosen).mean()),
        'oracle/mixed_mass': float(w.gather(0,chosen).mean()),
        'oracle/mean_nearest_angle_rms': float((info['leg_distance'].min(0).values/2).sqrt().mean()),
        'oracle/selected_angle_rms': float((info['leg_distance'].gather(0,chosen)/2).sqrt().mean())}
    if info['squared_terms'] is not None:
        terms = info['squared_terms']
        scales = geometry['scales_mean_squared']
        polar_only = geometry['geometry'] == POLAR_GEOMETRY
        names = ('signed_polar',) if polar_only else ('two_leg','pair_axis','signed_polar')
        for j,name in enumerate(names):
            metrics[f'oracle/{name}_normalized_selected'] = float((terms[...,j]/scales[j]).gather(0,chosen).mean())
        label = 'polar' if polar_only else 'joint'
        metrics[f'oracle/{label}_vs_nearest_switch_fraction'] = float((info['best'] != info['leg_distance'].argmin(0)).float().mean())
        if polar_only:
            metrics['oracle/selected_signed_polar_error_rms_rad'] = float(terms[...,0].gather(0,chosen).mean().sqrt())
    if geometry.get('geometry') == MOMENTUM_GEOMETRY:
        metrics['oracle/momentum_normalized_selected'] = float(info['distance'].gather(0,chosen).mean())
        metrics['oracle/momentum_vs_nearest_switch_fraction'] = float(
            (info['best'] != info['leg_distance'].argmin(0)).float().mean())
    if geometry.get('geometry') == RECO_MC_GEOMETRY:
        metrics['oracle/reconstructed_mc_p4_normalized_selected'] = float(info['distance'].gather(0,chosen).mean())
        metrics['oracle/reconstructed_mc_vs_nearest_switch_fraction'] = float(
            (info['best'] != info['leg_distance'].argmin(0)).float().mean())
    return reward, metrics
