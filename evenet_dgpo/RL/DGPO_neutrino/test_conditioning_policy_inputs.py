"""Exercise the production input-pruning paths, not just the standalone branch."""
import copy

import pytest
import torch
from torch import nn

from RL.DGPO_neutrino import dgpo_trainer as trainer
from RL.DGPO_neutrino.dgpo_utils import build_dgpo_loss, repeat_batch_for_candidates
from RL.DGPO_neutrino.sampling import generate_neutrino_candidates
from evenet.network.body.normalizer import Normalizer
from evenet.network.body.reward_conditioning import RewardConditioningProbe
from evenet.network.evenet_model import EveNetModel
from evenet.utilities.diffusion_sampler import DDIMSampler


VISIBLE_KEYS = tuple(f"lead_{leg}_visible_{axis}"
                     for leg in ("a", "b") for axis in ("px", "py", "pz"))


def batch():
    torch.manual_seed(41)
    result = dict(x=torch.randn(5, 4, 7), x_mask=torch.ones(5, 4, dtype=torch.bool),
                  conditions=torch.randn(5, 2), conditions_mask=torch.ones(5, 2, dtype=torch.bool),
                  x_invisible=torch.full((5, 2, 2), float('nan')),
                  x_invisible_mask=torch.ones(5, 2, dtype=torch.bool),
                  truth_tau_a_p4=torch.full((5, 4), float('nan')),
                  lead_a_truth_px=torch.full((5,), float('nan')))
    result.update({key: torch.arange(1., 6.) + index * .2 for index, key in enumerate(VISIBLE_KEYS)})
    return result


class ProbePolicy(nn.Module):
    """Small fixture using the actual EveNet probe/normalizer and native DDIM."""
    invisible_input_dim = 2
    invisible_padding = 0

    def __init__(self, basis):
        super().__init__()
        self.invisible_normalizer = Normalizer(torch.zeros(2), torch.ones(2) * .1,
                                               torch.ones(2, dtype=torch.bool), inv_cdf_index=[1])
        self.TruthGeneration = nn.Module()
        conditioning = self.TruthGeneration.visible_conditioning = nn.Module()
        conditioning.reward_probe = RewardConditioningProbe(7, basis=basis, width=8)
        conditioning.diagnostics = {}
        self.scale = nn.Parameter(torch.tensor(.1))

    def predict_diffusion_vector(self, *, noise_x, cond_x, time, mode, noise_mask):
        assert mode == 'neutrino'
        # Clean invisible values are deliberately NaN; only shape/mask are used.
        return EveNetModel._apply_reward_conditioning_probe(
            self, self.scale * noise_x, noise_x, cond_x, time,
            noise_mask, cond_x['x'], cond_x['x_mask'])


@pytest.mark.parametrize('basis', ['individual', 'relative'])
@pytest.mark.parametrize('parallel', [1, 2, 4])
def test_native_pruned_rollouts_preserve_visible_context(basis, parallel):
    data, model = batch(), ProbePolicy(basis)
    thin = trainer._dgpo_policy_conditioning_batch(data)
    assert 'truth_tau_a_p4' not in thin and 'lead_a_truth_px' not in thin
    for key in VISIBLE_KEYS:
        assert thin[key] is data[key]
    expanded = (repeat_batch_for_candidates(thin, 4, tensor_keys=trainer._DGPO_POLICY_BATCH_TENSOR_KEYS)
                if parallel == 4 else None)
    torch.manual_seed(42)
    actual = generate_neutrino_candidates(model, thin, DDIMSampler(torch.device('cpu')),
        K=4, num_ddim_steps=3, device=torch.device('cpu'), parallel_chains=parallel,
        expanded_batch=expanded)
    torch.manual_seed(42)
    full = generate_neutrino_candidates(model, data, DDIMSampler(torch.device('cpu')),
        K=4, num_ddim_steps=3, device=torch.device('cpu'), parallel_chains=parallel)
    assert actual.shape == (4, 5, 2, 2)
    assert torch.isfinite(actual).all()
    assert torch.equal(actual, full)


@pytest.mark.parametrize('basis', ['individual', 'relative'])
def test_native_microbatch_candidate_time_expansion_and_backward(basis):
    data, model = batch(), ProbePolicy(basis)
    reference = copy.deepcopy(model).requires_grad_(False)
    reference.TruthGeneration.visible_conditioning.reward_probe = None
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    branch = model.TruthGeneration.visible_conditioning.reward_probe
    for update in range(2):
        opt.zero_grad()
        for start, stop in ((0, 3), (3, 5)):
            micro = trainer._slice_event_batch(data, start, stop, batch_size=5)
            candidates = torch.randn(4, stop-start, 2, 2) * .1
            result = trainer.policy_evaluation_step(model, reference, micro, candidates,
                K=4, shared_noise=True, device=torch.device('cpu'), dtype=torch.float32,
                num_timesteps=2)
            current, ref, _, v, ref_v, mask, _, _, _, tiled, _ = result
            assert 'lead_a_truth_px' not in tiled and 'truth_tau_a_p4' not in tiled
            for key in VISIBLE_KEYS:
                assert torch.equal(tiled[key], data[key][start:stop].repeat(2*4))
            advantage = torch.tensor([-1., -.3, .3, 1.])[:, None].expand(4, stop-start)
            losses = [build_dgpo_loss(current[t], ref[t], advantage, 1., 4)[0] for t in range(2)]
            loss = sum(losses) / 2 + ((v-ref_v).square()*mask).mean()
            (loss * ((stop-start)/5)).backward()
        assert torch.isfinite(loss)
        assert branch.output.weight.grad.abs().sum() > 0
        if update:
            assert branch.encoder[0].weight.grad.abs().sum() > 0
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        opt.step()
    assert branch.output.weight.count_nonzero() > 0
