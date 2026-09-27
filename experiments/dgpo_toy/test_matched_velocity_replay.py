import pytest
import torch

from experiments.dgpo_toy.matched_velocity_replay import interpolate_state, solve_radius, velocity_distance


def test_interpolation_endpoints_exact_and_nonmutating():
    a={'w':torch.tensor([1.,2.]),'n':torch.tensor(2)}
    b={'w':torch.tensor([5.,6.]),'n':torch.tensor(2)}
    assert torch.equal(interpolate_state(a,b,0)['w'],a['w'])
    assert torch.equal(interpolate_state(a,b,1)['w'],b['w'])
    assert torch.equal(interpolate_state(a,b,.5)['w'],torch.tensor([3.,4.]))
    assert torch.equal(a['w'],torch.tensor([1.,2.]))
    with pytest.raises(ValueError):interpolate_state(a,b,1.1)
    with pytest.raises(ValueError):interpolate_state(a,{**b,'n':torch.tensor(3)},.5)


def test_radius_matches_quadratic_without_reward():
    r=solve_radius(lambda x:4*x*x,1.)
    assert abs(r['fraction']-.5)<1e-6 and r['scan_monotonic']
    with pytest.raises(ValueError):solve_radius(lambda x:x*x,2.)
    with pytest.raises(ValueError):solve_radius(lambda x:float('nan'),1.)


def test_first_crossing_does_not_require_global_monotonicity():
    def f(x):return 4*x if x<=.5 else 2-.5*(x-.5)
    r=solve_radius(f,1.)
    assert abs(r['fraction']-.25)<1e-6 and not r['scan_monotonic']


def test_probe_distance_zero_and_mean_square():
    class M:
        def __call__(self,x,t,c):return x
    x=torch.ones(12,3);p=(x,torch.zeros(12),torch.zeros(12,1),x)
    assert velocity_distance(M(),p)==0
    assert velocity_distance(M(),(*p[:3],torch.zeros_like(x)))==1
