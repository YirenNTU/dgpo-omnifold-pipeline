import copy
from pathlib import Path
import tempfile

import pytest
import torch

from experiments.dgpo_toy import closed_loop_lab as lab
from experiments.dgpo_toy import conditional as native
from experiments.dgpo_toy.test_saved_modulation_control import setup_control


def test_fixed_gain_calibration_uses_only_initial_condition_features():
    torch.set_num_threads(1)
    cfg=native.Config(dimensions=3,context_dim=1,hidden=8,ddim_steps=4)
    source=native.Denoiser(cfg)
    spec={'basis':'fourier','modulation':'film','nonlinear':True,'normalization':False,'layers':3,'width':32}
    rng=torch.random.get_rng_state().clone()
    calibration=lab.initial_encoder_rms_gain(cfg,source.state_dict(),seed=17,spec=spec,grid=64)
    assert torch.equal(rng,torch.random.get_rng_state())
    assert calibration['gain']>0 and not calibration['uses_truth_or_reward']
    with torch.random.fork_rng():
        torch.manual_seed(17)
        model=lab.ConditioningAblation(cfg,source.state_dict(),**spec,encoder_gain=calibration['gain'])
    c=((torch.arange(64)+.5)/64*2-1)[:,None]
    _,z=model.modulation(c,torch.zeros(64))
    assert abs(float(z.square().mean().sqrt())-1)<1e-6
    x,t=torch.randn(64,3),torch.rand(64)
    assert torch.equal(source(x,t,c),model(x,t,c))
    assert torch.equal(native.ddim(source,c,x,4),native.ddim(model,c,x,4))
    # Calibration is not recomputed when weights change; no event normalization.
    gain=model.encoder_gain
    with torch.no_grad():model.condition_input.weight.mul_(2)
    assert model.encoder_gain==gain


def test_gain_control_requires_everything_else_unchanged():
    with tempfile.TemporaryDirectory(dir=lab.ARTIFACTS,prefix='gain_contract_') as tmp:
        p,r,s=setup_control(Path(tmp))
        p['conditioning']['shift']['modulation']='film'
        p['conditioning']['shift']['encoder_gain']=10.
        p['external_control']['changed_factor']='encoder_gain'
        lab.validate_external_control(p)
        p['conditioning']['shift']['normalization']=False
        with pytest.raises(ValueError):lab.validate_external_control(p)


def test_degenerate_or_adaptive_calibration_rejected():
    cfg=native.Config(dimensions=3,context_dim=1,hidden=8)
    source=native.Denoiser(cfg)
    with pytest.raises(ValueError):
        lab.initial_encoder_rms_gain(cfg,source.state_dict(),seed=17,spec={'normalization':True})
    for gain in (0.,-1.,float('nan'),float('inf')):
        with pytest.raises(ValueError):lab.ConditioningAblation(cfg,source.state_dict(),encoder_gain=gain)
