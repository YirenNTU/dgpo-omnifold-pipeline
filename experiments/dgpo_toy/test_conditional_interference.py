import copy
import torch
from experiments.dgpo_toy.conditional_interference import derivative_matrix, displaced, flat_grad


def test_transfer_orientation():
    reward=torch.eye(2)
    loss=torch.tensor([[-1.,2.],[3.,-4.]])
    torch.testing.assert_close(derivative_matrix(reward,loss),torch.tensor([[1.,-3.],[-2.,4.]],dtype=torch.float64))


def test_displacement_does_not_mutate_source():
    model=torch.nn.Linear(2,1)
    before=copy.deepcopy(model.state_dict())
    direction=torch.ones(3)
    after=displaced(model,direction,.2)
    for k,v in before.items():
        torch.testing.assert_close(model.state_dict()[k],v,rtol=0,atol=0)
        torch.testing.assert_close(after.state_dict()[k],v+.2)


def test_reward_gradient_matches_finite_difference():
    model=torch.nn.Linear(2,1,dtype=torch.float64)
    x=torch.tensor([[.3,.4]],dtype=torch.float64)
    gradient=flat_grad(model(x).square().mean(),model)
    direction=torch.tensor([.1,.2,.3],dtype=torch.float64)
    scale=1e-5
    actual=(displaced(model,direction,scale)(x).square().mean()-displaced(model,direction,-scale)(x).square().mean())/(2*scale)
    torch.testing.assert_close(actual,gradient@direction)
