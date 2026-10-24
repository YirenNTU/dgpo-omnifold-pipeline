"""Opt-in proper-score continuation of the *unchanged* tau DDIM generator.

This is supervised pathwise training, not DGPO or a fitted reward. The
off-diagonal U statistic preserves the population Energy score for finite K.
"""
from contextlib import contextmanager
from functools import partial
import math
import torch


def resumed_lr_multiplier(step, *, pivot, reference_steps, new_steps, epochs,
                          warmup_epochs, warm_up, floor):
    """Keep the saved LR, then traverse the old schedule at the new epoch rate."""
    coordinate = pivot + (step-pivot)*(reference_steps/new_steps)
    warmup = warmup_epochs*reference_steps if warm_up else 0
    total = epochs*reference_steps
    if coordinate < warmup:
        return max(0., coordinate)/max(1., warmup)
    progress = min(1., max(0., (coordinate-warmup)/max(1., total-warmup)))
    return floor+(1-floor)*.5*(1+math.cos(math.pi*progress))


def energy_score(samples, truth):
    if samples.ndim != 3 or truth.shape != (samples.shape[0], samples.shape[2]) or samples.shape[1] < 2:
        raise ValueError('Require B x K>=2 x D samples and B x D truth')
    if not torch.isfinite(samples).all() or not torch.isfinite(truth).all():
        raise ValueError('Nonfinite energy-score inputs')
    k = samples.shape[1]
    attraction = torch.linalg.vector_norm(samples-truth[:, None], dim=-1).mean(1)
    distance = torch.linalg.vector_norm(samples[:, :, None]-samples[:, None, :], dim=-1)
    repulsion = distance.sum((1, 2))/(2*k*(k-1))
    # sqrt(D) changes units only, not the optimum or attraction/repulsion ratio.
    scale = samples.shape[-1]**.5
    return (attraction-repulsion)/scale, attraction/scale, repulsion/scale


def validate(spec):
    if set(spec) != {'enabled', 'candidates', 'ddim_steps', 'checkpoint_steps', 'validation_batches', 'seed', 'observable'}:
        raise ValueError('Unexpected EnergyScore configuration')
    if spec['enabled'] is not True or spec['checkpoint_steps'] is not True:
        raise ValueError('Require explicitly enabled, checkpointed EnergyScore')
    for key in ('candidates', 'ddim_steps', 'validation_batches', 'seed'):
        if type(spec[key]) is not int or spec[key] < 1:
            raise ValueError('Positive integer required: '+key)
    if spec['candidates'] < 2 or spec['ddim_steps'] != 20:
        raise ValueError('Require K>=2 and unchanged DDIM20')
    if spec['observable'] != 'acoplanarity':
        raise ValueError('Require observed lead momenta for acoplanarity')


def acoplanarity(offsets, visible):
    """Uncalibrated lab directions: never force a back-to-back tau pair."""
    phis = []
    for j, leg in enumerate('ab'):
        p = visible[..., j, :]
        theta = torch.atan2(torch.linalg.vector_norm(p[..., :2], dim=-1), p[..., 2]) + offsets[..., j, 0]
        phi = torch.atan2(p[..., 1], p[..., 0]) + offsets[..., j, 1]
        # Cartesian extension agrees with the historical generated directions
        # even if a predicted theta crosses a pole; do not clip or drop samples.
        phis.append(torch.atan2(theta.sin()*phi.sin(), theta.sin()*phi.cos()))
    difference = phis[0]-phis[1]
    return torch.pi-torch.atan2(difference.sin(), difference.cos()).abs()


VISIBLE_COLUMNS = [f'lead_{leg}_visible_{axis}' for leg in 'ab' for axis in ('px','py','pz')]


def visible_momenta(batch):
    # lead_a_px is the leading charged object, NOT always total reconstructed
    # visible momentum. Use the same exported visible system as the targets.
    if any(k not in batch for k in VISIBLE_COLUMNS):
        raise ValueError('Require original reconstructed visible-system columns: '+str(VISIBLE_COLUMNS))
    visible = torch.stack([batch[k].reshape(-1) for k in VISIBLE_COLUMNS],-1).reshape(-1,2,3)
    if not torch.isfinite(visible).all() or (torch.linalg.vector_norm(visible[...,:2],dim=-1)<=0).any():
        raise ValueError('Invalid visible-system transverse momentum')
    return visible


@contextmanager
def inference_function(model):
    """Same deterministic function as generation; keep active through backward."""
    modes = [(m, m.training) for m in model.modules()]
    checkpoints = [(m, m.gradient_checkpointing) for m in model.modules() if hasattr(m, 'gradient_checkpointing')]
    try:
        model.eval()
        for m, _ in checkpoints:
            m.gradient_checkpointing = False  # DDIM already checkpoints each step
        yield
    finally:
        for m, old in checkpoints:
            m.gradient_checkpointing = old
        for m, old in modes:
            m.training = old


def endpoint_loss(model, batch, spec, *, differentiable):
    """Score physical acoplanarity in radians; no truth passed to denoiser."""
    from RL.DGPO_neutrino.dgpo_utils import repeat_batch_for_candidates
    from evenet.utilities.diffusion_sampler import DDIMSampler
    validate(spec)
    if 'classification' not in batch:
        raise ValueError('Actual process labels required; no label-zero fallback')
    truth = batch['x_invisible']
    mask = batch['x_invisible_mask']
    if truth.shape[1:] != (2, 2) or mask.shape != truth.shape[:2] or not mask.bool().all():
        raise ValueError('Require two valid legs with original theta/phi targets')
    normalizer = model.invisible_coordinate_normalizer(batch)
    if getattr(normalizer, 'inv_cdf_index', []):
        raise ValueError('Pilot requires the epoch287 affine angular normalizer')
    visible = visible_momenta(batch).to(truth)
    target = acoplanarity(truth, visible)[:, None]
    # Candidate inputs contain observed conditioning and shapes only.
    observed = {k: batch[k] for k in ('x','x_mask','conditions','conditions_mask','classification','x_invisible_mask')}
    observed['x_invisible'] = torch.zeros_like(truth)
    expanded = repeat_batch_for_candidates(observed, spec['candidates'])
    noise_mask = expanded['x_invisible_mask'][..., None]
    shape = (len(truth)*spec['candidates'], 2, model.invisible_input_dim)
    predictor = partial(model.predict_diffusion_vector, cond_x=expanded, mode='neutrino', noise_mask=noise_mask)
    sampler = DDIMSampler(truth.device, x0_mode='legacy')
    draws = sampler.sample(shape, predictor, num_steps=20, noise_mask=noise_mask,
        normalize_fn=None, differentiable=differentiable,
        checkpoint_steps=bool(differentiable and spec['checkpoint_steps']))
    physical = normalizer.denormalize_grad(draws, noise_mask, remove_padding=True)
    offsets = physical.reshape(spec['candidates'], len(truth), 2, 2).transpose(0, 1)
    samples = acoplanarity(offsets, visible[:, None])[..., None]
    score, attraction, repulsion = energy_score(samples, target)
    return score.mean(), dict(energy=score.mean(), attraction=attraction.mean(),
        repulsion=repulsion.mean(), spread=samples.std(1, unbiased=True).mean())


def engine_step(engine, batch, batch_idx, *, training):
    from evenet.utilities.fourier_integration import paired_diffusion_rng
    spec = engine.config.options.Training.get('EnergyScore')
    validate(spec)
    engine.current_schedule = dict(generation=False, neutrino_generation=False, deterministic=False)
    engine.current_stage = 'energy-score'
    if not training and batch_idx >= spec['validation_batches']:
        return None
    if engine.include_famo or engine.ema_model is not None or engine.apply_event_weight:
        raise ValueError('Energy pilot requires FAMO/EMA/event weighting disabled')
    if training:
        opts = engine.optimizers(); opts = list(opts) if isinstance(opts, (list, tuple)) else [opts]
        schedules = engine.lr_schedulers(); schedules = list(schedules) if isinstance(schedules, (list, tuple)) else [schedules]
        engine.current_step = int(schedules[0].state_dict().get('last_epoch', 0))
        for opt in opts:
            opt.zero_grad()
    inputs = {k: v.to(engine.device) if torch.is_tensor(v) else v for k, v in batch.items()}
    # Validation noise is frozen across epochs; training noise varies by epoch.
    with inference_function(engine.model), paired_diffusion_rng(spec['seed'], training=training,
            epoch=engine.current_epoch if training else 0, batch_idx=batch_idx,
            rank=engine.global_rank, device=engine.device):
        loss, metrics = endpoint_loss(engine.model, inputs, spec, differentiable=training)
        if training:
            engine.safe_manual_backward(loss)
            torch.nn.utils.clip_grad_norm_(engine.model.parameters(), 1., error_if_nonfinite=True)
            for opt in opts:
                opt.step()
            for scheduler in schedules:
                scheduler.step()
    prefix = 'train' if training else 'val'
    for key, value in metrics.items():
        engine.log(prefix+'/proper_score/'+key, value.detach(), sync_dist=True,
                   on_step=training, on_epoch=True, batch_size=len(inputs['x']))
    engine.log(prefix+'/loss', loss.detach(), sync_dist=True, on_step=training,
               on_epoch=True, batch_size=len(inputs['x']))
    return loss.detach()
