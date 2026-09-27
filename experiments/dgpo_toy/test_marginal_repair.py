import numpy as np
from scipy.stats import spearmanr
from experiments.dgpo_toy.marginal_repair import fit_maps, transport


def test_independent_normal_repair_and_ranks():
    rng=np.random.default_rng(17)
    maps=fit_maps(rng.normal(size=(20000,2))*3+8)
    z=rng.normal(size=(10000,2))*3+8
    fixed=transport(z,maps)
    assert np.max(np.abs(fixed.mean(0)))<.05
    assert np.max(np.abs(fixed.var(0)-1))<.08
    assert spearmanr(z[:,0],fixed[:,0]).statistic>.99999


def test_ties_and_out_of_range_are_finite():
    maps=fit_maps(np.ones((20,2)))
    fixed=transport(np.array([[-100.,0.],[1.,100.]]),maps)
    assert np.isfinite(fixed).all()
    assert np.array_equal(fixed,np.zeros((2,2)))
