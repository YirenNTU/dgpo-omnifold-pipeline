from types import SimpleNamespace

import numpy as np
import pytest
import torch

from evenet.utilities.joint_coverage import JointCoverageValidation, event_noise, due
from test_h4_ddim_coverage import panel_fixture


def test_noise_is_identical_across_sixteen_rank_partition_and_preserves_rng():
    state = torch.get_rng_state().clone()
    whole = event_noise(torch.arange(32), 4, 42)
    merged = torch.empty_like(whole)
    for rank in range(16):
        ids = torch.arange(rank, 32, 16)
        merged[ids] = event_noise(ids, 4, 42)
    assert torch.equal(whole, merged)
    assert torch.equal(state, torch.get_rng_state())
    assert [epoch + 1 for epoch in range(250) if due(epoch, 25)] == list(range(25, 251, 25))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_callback_baseline_replay_state_and_zero_update_comparison(tmp_path, monkeypatch, dtype):
    import RL.DGPO_neutrino.sampling as sampling
    panel = panel_fixture(8)
    path = tmp_path / 'panel.pt'
    torch.save(panel, path)
    config = dict(every_n_epochs=25, K=4, events=8, panel_path=str(path),
                  seed=42, batch_size=4, ddim_steps=20, x0_mode='legacy', bootstrap=20)
    callback = JointCoverageValidation(config, tmp_path)
    # Test callback lifecycle and real reconstruction/bootstrap; sampling is
    # independently covered by the production sampler regression tests.
    def fake_generate(model, batch, sampler, K, **kwargs):
        assert all(value.dtype == dtype for value in batch.values()
                   if isinstance(value, torch.Tensor) and value.is_floating_point())
        assert batch['x_invisible'].dtype == dtype
        torch.rand(3)  # Must not perturb subsequent training RNG.
        return torch.stack([sampler.prior_sde((len(batch['x_invisible']), 2, 2)) * .001 for _ in range(K)])
    monkeypatch.setattr(sampling, 'generate_neutrino_candidates', fake_generate)
    monkeypatch.setattr(callback, 'log_plots', lambda *args: None)
    logged = []
    logger = SimpleNamespace(log_metrics=lambda metrics, step: logged.append(metrics))
    model = torch.nn.Sequential(torch.nn.Dropout(), torch.nn.Linear(2, 2)).to(dtype=dtype)
    model.train()
    model[1].eval()
    module = SimpleNamespace(model=model, device=torch.device('cpu'))
    trainer = SimpleNamespace(global_rank=0, world_size=1, global_step=0,
                              current_epoch=0, sanity_checking=False, loggers=[logger])
    rng = torch.get_rng_state().clone()
    callback.on_train_start(trainer, module)
    saved = torch.load(tmp_path / 'joint_coverage/epoch-0000.pt', weights_only=True)
    assert saved['generated'].dtype == dtype
    assert torch.equal(torch.get_rng_state(), rng)
    assert model.training and not model[1].training
    trainer.current_epoch = 23
    callback.on_validation_epoch_end(trainer, module)
    assert len(logged) == 1
    trainer.current_epoch = 24
    callback.on_validation_epoch_end(trainer, module)
    assert len(logged) == 2
    assert logged[-1]['joint_coverage/paired/joint/0.0001/after_minus_before_draw_fraction'] == 0
    assert logged[-1]['joint_coverage/change_from_baseline/topology/joint_radius/w1_radians'] == 0
    callback.on_validation_epoch_end(trainer, module)
    assert len(logged) == 2
    restored = JointCoverageValidation(config, tmp_path)
    restored.load_state_dict(callback.state_dict())
    restored.on_train_start(trainer, module)
    assert len(logged) == 2
    assert np.array_equal(restored.baseline.numpy(), callback.baseline.numpy())
    with pytest.raises(ValueError, match='changed'):
        JointCoverageValidation({**config, 'K': 8}, tmp_path).load_state_dict(callback.state_dict())
