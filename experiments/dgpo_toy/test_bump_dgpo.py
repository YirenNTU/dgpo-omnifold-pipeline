import copy
import torch

from experiments.dgpo_toy.bump_dgpo import Policy, Ratio
from experiments.dgpo_toy import conditioning_placement as p, conditional as n
from experiments.dgpo_toy.conditioning_mse import mean_function


def test_wrapper_matches_flat_model_and_broadcasts():
    model = p.VelocityModel(8, 'early')
    policy = Policy(model)
    x, c, t = torch.randn(5,1), torch.rand(5,1), torch.rand(5)
    assert torch.equal(policy(x,t,c), model(x,t,c))
    assert policy(torch.randn(3,4,1), torch.tensor(.3), torch.rand(3,1,1)).shape == (3,4,1)


def test_ratio_includes_tail_bins_and_narrow_target_is_finite():
    ratio = Ratio(torch.ones(32,64,dtype=torch.int64), 'narrow_bump')
    y = torch.tensor([[-100.], [100.]])
    c = torch.zeros(2,1)
    assert ratio.ids(y,c)[1].tolist() == [0,63]
    assert torch.isfinite(ratio(y,c)).all()
    grid = torch.linspace(-1,1,10000)[:,None]
    assert abs(mean_function(grid,'narrow_bump').mean()) < .001
    assert abs(mean_function(grid,'narrow_bump').square().mean()-1) < .001


def test_native_loss_learns_zero_output_adapter_without_changing_reference():
    model = Policy(p.VelocityModel(8,'early'))
    reference = copy.deepcopy(model).requires_grad_(False)
    reward = Ratio(torch.ones(32,64,dtype=torch.int64))
    cfg = n.Config(timesteps=2,ddim_steps=2)
    loss,_ = n.dgpo_objective(model,reference,reward,None,cfg,
        torch.rand(3,1),torch.randn(3,4,1),torch.rand(2,3),
        torch.randn(2,3,1),velocity_coefficient=1.)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.model.adapter[-1].weight.grad.norm()>0
    assert all(v.grad is None for v in reference.parameters())


def test_fine_bins_preserve_coarse_truth_masses_and_smoothing():
    fine = Ratio(torch.ones(128,64,dtype=torch.int64), 'narrow_bump')
    coarse = Ratio(torch.full((32,64),4,dtype=torch.int64), 'narrow_bump')
    # Uniform q in both, so exp(log ratio)/64 recovers truth bin mass.
    fine_p = fine.log_ratio.double().exp()/64
    coarse_p = coarse.log_ratio.double().exp()/64
    torch.testing.assert_close(fine_p.reshape(32,4,64).mean(1),coarse_p,atol=1e-7,rtol=1e-5)
    ci,_=fine.ids(torch.zeros(2,1),torch.tensor([[-1.],[1.]]))
    assert ci.tolist()==[0,127]
