import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from evenet.network.body.reward_conditioning import RewardConditioningProbe
from evenet.network.body.visible_conditioning import visible_conditioning_spec
from RL.DGPO_neutrino.conditioning_probe import extend_optimizer_for_probe, PairedUpdateCache


def inputs():
    return dict(tokens=torch.randn(5, 4, 7), visible_mask=torch.ones(5, 4, 1),
        noisy=torch.randn(5, 2, 2), physical_noisy_delta=torch.randn(5, 2, 2)*.03,
        time=torch.rand(5), invisible_mask=torch.ones(5, 2, 1),
        batch={f'lead_{leg}_visible_{axis}': torch.randn(5) for leg in ('a', 'b') for axis in ('px', 'py', 'pz')})


def test_identical_capacity_zero_outputs_and_rng_unchanged():
    state = torch.get_rng_state().clone()
    a, b = (RewardConditioningProbe(7, basis=x) for x in ('individual', 'relative'))
    assert torch.equal(state, torch.get_rng_state())
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())
    assert all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict())
    data = inputs()
    for model in (a, b):
        assert torch.count_nonzero(model(**data)) == 0
    assert not torch.equal(a.angle_features(data['physical_noisy_delta'], data['batch']),
                           b.angle_features(data['physical_noisy_delta'], data['batch']))


def test_no_clean_truth_leakage_masking_and_gradients():
    model = RewardConditioningProbe(7, basis='relative', width=16)
    data = inputs()
    model(**data).sum().backward()
    assert model.output.weight.grad.abs().sum() > 0
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    opt.step(); opt.zero_grad()
    original = model(**data)
    data['batch']['x_invisible'] = torch.full((5, 2, 2), float('nan'))
    data['batch']['truth_tau_a_p4'] = torch.full((5, 4), float('nan'))
    assert torch.equal(original, model(**data))
    model(**data).square().sum().backward()
    assert model.encoder[0].weight.grad.abs().sum() > 0
    data['visible_mask'][0] = 0
    data['tokens'][0] = float('nan')
    data['invisible_mask'][1] = 0
    data['noisy'][1] = float('nan')
    data['physical_noisy_delta'][1] = float('nan')
    for key, value in data['batch'].items():
        if key.startswith('lead_'):
            value[:2] = float('nan')
    out = model(**data)
    assert torch.isfinite(out).all()
    assert torch.count_nonzero(out[:2]) == 0
    model.zero_grad()
    out.square().sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_classifier_spec_is_unchanged():
    cfg = SimpleNamespace(VisibleConditioning={'classifier_enabled': True, 'diffusion_enabled': True,
        'diffusion_reward_probe': {'enabled': True, 'basis': 'relative'}})
    kwargs = dict(feature_names=['Part_eta', 'Part_phi'], token_dim=7, hidden_dim=16,
                  num_layers=3, n_branches=2, output_dim=2)
    spec = visible_conditioning_spec(cfg, target='classifier', **kwargs)
    assert 'reward_probe' not in spec
    assert 'reward_probe' in visible_conditioning_spec(cfg, target='diffusion', **kwargs)


class Core(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Linear(3, 3)
        self.TruthGeneration = nn.Module()
        self.TruthGeneration.visible_conditioning = nn.Module()
        self.TruthGeneration.visible_conditioning.old = nn.Linear(3, 3)

    def opt(self):
        return torch.optim.AdamW([
            {'params': list(self.body.parameters()), 'group_name': 'body', 'lr': 5e-5},
            {'params': list(self.TruthGeneration.parameters()), 'group_name': 'visible_conditioning', 'lr': 1e-4}])


def test_late_optimizer_migration_keeps_old_moments_clocks_and_reference():
    model = Core()
    old_opt = model.opt()
    sum(p.sum() for p in model.parameters()).backward()
    old_opt.step()
    old_opt.zero_grad()
    ckpt = {'state_dict': copy.deepcopy(model.state_dict()),
            'dgpo_optimizer_state_dict': {'optimizer': copy.deepcopy(old_opt.state_dict()),
                'scheduler': {'last_epoch': 1130, 'base_lrs': [5e-5, 1e-4]}, 'lr_schedule': {'total_steps': 15000}},
            'dgpo_round_ref_state_dict': {'ref': torch.ones(3)}}
    old_params = dict(model.named_parameters())
    branch = RewardConditioningProbe(7, basis='relative', width=16)
    model.TruthGeneration.visible_conditioning.reward_probe = branch
    new_opt = model.opt()
    migrated, count = extend_optimizer_for_probe(model, new_opt, ckpt)
    assert count == len(list(branch.parameters()))
    assert migrated['dgpo_round_ref_state_dict'] is ckpt['dgpo_round_ref_state_dict']
    assert migrated['dgpo_optimizer_state_dict']['scheduler'] is ckpt['dgpo_optimizer_state_dict']['scheduler']
    new_opt.load_state_dict(migrated['dgpo_optimizer_state_dict']['optimizer'])
    for p in old_params.values():
        for k, v in old_opt.state[p].items():
            assert torch.equal(v, new_opt.state[p][k])
    for p in branch.parameters():
        assert p not in new_opt.state
    assert [g['lr'] for g in new_opt.param_groups] == [5e-5, 1e-4]
    with torch.no_grad():
        model.body.weight.add_(1)
    with pytest.raises(ValueError, match='not restored exactly'):
        extend_optimizer_for_probe(model, new_opt, ckpt)


def test_cache_pairs_data_and_rng(tmp_path):
    cfg = dict(update_cache_directory=str(tmp_path), update_cache_mode='write',
               relative_steps=[0, 1, 5, 20, 35, 50], update_seed=13)
    a = PairedUpdateCache(cfg, 2, 1130)
    data = {'x': torch.randn(5, 2), 'extra': torch.arange(5)}
    a.before_update(data)
    noise = torch.randn(20)
    b = PairedUpdateCache({**cfg, 'update_cache_mode': 'read'}, 2, 1130)
    data2 = next(b.iterator(None, {}))
    assert all(torch.equal(data[k], data2[k]) for k in data)
    torch.randn(99)
    b.before_update(data2)
    assert torch.equal(noise, torch.randn(20))


def test_real_model_probe_helper_uses_normalizer_and_not_truth():
    from evenet.network.evenet_model import EveNetModel
    from evenet.network.body.normalizer import Normalizer
    data = inputs()
    conditioning = SimpleNamespace(reward_probe=RewardConditioningProbe(7, basis='relative', width=8), diagnostics={})
    core = SimpleNamespace(TruthGeneration=SimpleNamespace(visible_conditioning=conditioning),
        invisible_normalizer=Normalizer(torch.tensor([.1, -.2]), torch.tensor([.03, .02]),
                                        torch.ones(2, dtype=torch.bool), inv_cdf_index=[1], padding_size=5))
    vector = torch.randn_like(data['noisy'])
    actual = EveNetModel._apply_reward_conditioning_probe(core, vector, data['noisy'], data['batch'],
        data['time'], data['invisible_mask'], data['tokens'], data['visible_mask'])
    assert torch.equal(actual, vector)
    assert torch.isfinite(actual).all()


def test_new_policy_branch_does_not_modify_serialized_reference():
    from RL.DGPO_neutrino.model_utils import _match_saved_reference_token_readout, state_dict_sha256
    model = Core()
    saved = copy.deepcopy(model.state_dict())
    saved_digest = state_dict_sha256(model)
    model.TruthGeneration.visible_conditioning.spec = {}
    model.TruthGeneration.visible_conditioning.reward_probe = RewardConditioningProbe(7, basis='relative', width=8)
    _match_saved_reference_token_readout(model, saved)
    model.load_state_dict(saved, strict=True)
    assert model.TruthGeneration.visible_conditioning.reward_probe is None
    assert state_dict_sha256(model) == saved_digest
