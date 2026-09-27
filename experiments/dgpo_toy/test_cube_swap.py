import torch
from experiments.dgpo_toy.cube_swap import swaps,modes,truth_shape
from experiments.dgpo_toy.nonperiodic_cube import Data
from experiments.dgpo_toy import conditional as native


def test_truth_shape_preserves_every_mode():
    data=Data(native.Config(dimensions=3,context_dim=1))
    ids=torch.arange(8).repeat(100)
    y=truth_shape(ids,data.centers,native.generator(17))
    assert torch.equal(modes(y),ids)


def test_swap_modes_and_condition_are_preserved(monkeypatch):
    data=Data(native.Config(dimensions=3,context_dim=1))
    def fake(model,c,z,steps):return z.sign()*(1+.02*z.abs())
    monkeypatch.setattr(native,'ddim',fake)
    panels,health=swaps(None,data,8,128,1024,17)
    assert health['min_cell_count']>=16
    assert torch.equal(modes(panels['A']['negative']),modes(panels['C']['negative']))
    assert torch.equal(modes(panels['B']['negative']),modes(panels['D']['negative']))
    for p in panels.values():
        assert torch.equal(p['c'],panels['A']['c'])
        assert torch.equal(p['positive'],panels['A']['positive'])
    # Each generated shape is selected inside its actual grid cell and mode.
    assert (panels['C']['negative'].abs()>=1).all()


def test_missing_mode_stops_instead_of_mixing(monkeypatch):
    import pytest
    data=Data(native.Config(dimensions=3,context_dim=1))
    monkeypatch.setattr(native,'ddim',lambda model,c,z,steps:torch.ones_like(z))
    with pytest.raises(ValueError,match='Sparse'):
        swaps(None,data,8,128,1024,17)
