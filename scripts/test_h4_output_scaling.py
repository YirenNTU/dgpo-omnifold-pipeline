import copy
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_h4_output_scaling as launcher
from RL.DGPO_neutrino.omnifold_ztautau.feature_standardization import FixedFeatureStandardizer, fit_output_standardizer
from RL.DGPO_neutrino.omnifold_ztautau.stage import build_fit_config


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.bank = nn.Module()
        self.bank.topology_conditioning = self.bank.topology_direct_logit = False
        self.bank.topology_encoder = nn.Sequential(nn.Linear(4, 4), nn.GELU(), nn.Dropout(.25))
        self.bank.topology_standardizer = FixedFeatureStandardizer(4)

    def forward(self, c, z):
        return self.bank.topology_standardizer(self.bank.topology_encoder(z)).sum(1)


def population():
    g = torch.Generator().manual_seed(14)
    return [(torch.randn(64,4,generator=g), torch.randn(64,4,generator=g) + k, torch.ones(64)) for k in (0,1)]


@pytest.mark.parametrize('arm', ['control','standardized'])
def test_contract(arm):
    cfg = launcher.validated_config(arm)
    fit = build_fit_config(cfg['dgpo']['adaptive_omnifold']['audit_fit'], n_train=20000, n_validation=4000)
    fit.validate()
    assert fit.steps == fit.min_steps == 300
    assert fit.fourier_output_standardization == (arm == 'standardized')
    assert fit.checkpoint_selection_metric == 'loss'
    assert cfg['platform']['number_of_workers'] == 16


def test_reject_lr_change():
    cfg = launcher.validated_config('standardized')
    cfg['dgpo']['adaptive_omnifold']['audit_fit']['learning_rate'] = .1
    with patch.object(launcher, 'read_overlay_yaml', return_value=cfg):
        with pytest.raises(ValueError, match='Only output'):
            launcher.validated_config('standardized')


def test_long_fit_preserves_intervention_and_runtime_contract():
    cfg = launcher.validated_config('standardized-long')
    runtime_guard(cfg)
    fit = build_fit_config(cfg['dgpo']['adaptive_omnifold']['audit_fit'], n_train=20000, n_validation=4000)
    fit.validate()
    assert (fit.steps, fit.min_steps) == (3000, 1000)
    assert fit.restore_best and fit.fourier_output_standardization
    assert fit.checkpoint_selection_metric == 'loss'
    assert fit.lr_scheduler == 'constant'
    assert cfg['platform']['number_of_workers'] == 16
    retry = launcher.retry_config(cfg, 'standardized-long', 'retry1')
    assert retry['logger']['wandb']['id'] == 'h4scllong1-retry1'
    assert 'h4_scale_standardized_long-retry1/' in retry['nersc']['ray']['results_dir']
    runtime_guard(retry)


@pytest.mark.parametrize('key,value', [('steps', 300), ('min_steps', 0),
    ('validation_patience_epochs', 1), ('fourier_output_standardization', False),
    ('checkpoint_selection_metric', 'auc'), ('lr_scheduler', 'warmup_cosine')])
def test_long_runtime_rejects_changed_contract(key, value):
    cfg = launcher.validated_config('standardized-long')
    cfg['dgpo']['adaptive_omnifold']['audit_fit'][key] = value
    with pytest.raises(ValueError, match='Long scaling fit'):
        runtime_guard(cfg)


def runtime_guard(cfg):
    import ast
    from types import SimpleNamespace
    source = launcher.ROOT / 'evenet_dgpo/RL/DGPO_neutrino/dgpo_trainer.py'
    tree = ast.parse(source.read_text())
    branch = next(n for n in ast.walk(tree) if isinstance(n,ast.If)
                  and isinstance(n.test,ast.Name) and n.test.id == 'classifier_only')
    index = next(i for i,n in enumerate(branch.body) if isinstance(n,ast.Assign)
                 and any(isinstance(t,ast.Name) and t.id == 'classifier_lr_replay' for t in n.targets))
    namespace = dict(_dgpo_cfg_get=lambda c,k,d:c.get(k,d),
        global_config=SimpleNamespace(experiment=cfg['experiment']),
        adaptive_cfg=SimpleNamespace(audit_fit=cfg['dgpo']['adaptive_omnifold']['audit_fit']))
    exec(compile(ast.Module(body=branch.body[index:index+3],type_ignores=[]),'<actual-runtime-guard>','exec'),namespace)
    assert namespace['classifier_lr_replay']  # Preserve identity-based split/cache path.


@pytest.mark.parametrize('arm',['control','standardized'])
def test_actual_runtime_guard_accepts_300_step_arms(arm):
    cfg = launcher.validated_config(arm)
    runtime_guard(cfg)
    retry = launcher.retry_config(cfg,arm,'retry1')
    runtime_guard(retry)
    assert retry['logger']['wandb']['id'].endswith('-retry1')
    assert retry['options']['Training']['model_checkpoint_load_path'] == cfg['options']['Training']['model_checkpoint_load_path']
    assert '-retry1/' in retry['dgpo']['adaptive_omnifold']['audit_fit']['representation_export_dir']


def test_runtime_guard_still_rejects_wrong_budget_and_unknown_protocol():
    cfg = launcher.validated_config('control')
    cfg['dgpo']['adaptive_omnifold']['audit_fit']['steps'] = 301
    with pytest.raises(ValueError,match='exactly 300'):
        runtime_guard(cfg)
    cfg = launcher.validated_config('control')
    cfg['experiment']['protocol'] = 'unknown'
    with pytest.raises(ValueError,match='minimum 1000'):
        runtime_guard(cfg)


def test_fixed_statistics_rng_modes_and_roundtrip():
    torch.manual_seed(6)
    model = Toy()
    pools = population()
    model.eval()
    with torch.no_grad():
        features = torch.cat([model.bank.topology_encoder(z) for _,z,_ in pools])
    model.train()
    state = torch.get_rng_state().clone()
    old_parameters = {n: p.clone() for n,p in model.named_parameters()}
    fit_output_standardizer(model, pools)
    assert torch.equal(state, torch.get_rng_state()) and model.training
    for n,p in model.named_parameters():
        assert torch.equal(p, old_parameters[n])
    normalized = model.bank.topology_standardizer(features)
    assert torch.allclose(normalized.mean(0), torch.zeros(4), atol=2e-6)
    assert torch.allclose(normalized.std(0,unbiased=False), torch.ones(4), atol=2e-6)
    saved = copy.deepcopy(model.state_dict())
    restored = Toy()
    restored.load_state_dict(saved)
    model.eval(); restored.eval()
    c,z,_ = pools[0]
    assert torch.equal(model(c,z), restored(c,z))
    legacy = {k:v for k,v in saved.items() if 'topology_standardizer' not in k}
    restored.load_state_dict(legacy, strict=True)
    assert torch.equal(restored.bank.topology_standardizer.scale, torch.ones(4))


def _worker(rank, rendezvous):
    torch.set_num_threads(1)
    torch.distributed.init_process_group('gloo',init_method='file://' + rendezvous,rank=rank,world_size=2)
    try:
        torch.manual_seed(6)
        model = Toy()
        reference = copy.deepcopy(model)
        pools = population()
        fit_output_standardizer(reference, pools)
        fit_output_standardizer(model, pools, rank=rank, world=2)
        assert torch.allclose(model.bank.topology_standardizer.mean, reference.bank.topology_standardizer.mean)
        assert torch.allclose(model.bank.topology_standardizer.scale, reference.bank.topology_standardizer.scale)
    finally:
        torch.distributed.destroy_process_group()


def test_two_rank_moments(tmp_path):
    torch.multiprocessing.spawn(_worker,args=(str(tmp_path/'rendezvous'),),nprocs=2,join=True)


def test_real_classifier_fit_and_fixed_buffer_persistence():
    from RL.DGPO_neutrino.omnifold_ztautau.test_ztautau_omnifold import _FakeZtautauBackbone, _event_batch_with_pair_context
    from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import EvenetAdapterRatioClassifier, pack_event_inputs
    from RL.DGPO_neutrino.omnifold_ztautau.ratio_fit import RatioFitConfig, fit_density_ratio
    torch.set_num_threads(1)
    torch.manual_seed(55)
    packed, spec = pack_event_inputs(_event_batch_with_pair_context(batch_size=32), include_pairwise_context=True)
    model = EvenetAdapterRatioClassifier(_FakeZtautauBackbone(), spec,
        train_backbone=True, decoder_hidden_dim=8, decoder_layers=1, decoder_heads=2,
        adapter_bottleneck=4, head_dropout=.2, periodic_pair_features=True,
        topology_fourier_embedding=True, topology_hidden_dim=8, topology_embedding_dim=4,
        topology_fusion_hidden_dim=8)
    data = (packed,torch.randn(32,4),torch.ones(32),packed,torch.randn(32,4),torch.ones(32))
    fit_output_standardizer(model, [data[:3],data[3:]])
    expected = copy.deepcopy(model.bank.topology_standardizer.state_dict())
    cfg = RatioFitConfig(steps=3,batch_size=16,learning_rate=2e-4,restore_best=False,
        fourier_output_standardization=True, validation_interval_steps=1,
        validation_patience_evaluations=10)
    result = fit_density_ratio(model,*data,cfg,42,validation=data)
    assert result.steps_completed == 3
    for key,value in expected.items():
        assert torch.equal(value,model.bank.topology_standardizer.state_dict()[key])
    assert not torch.equal(expected['scale'],torch.ones(4))
