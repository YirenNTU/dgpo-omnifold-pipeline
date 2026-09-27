from pathlib import Path
import sys
from unittest.mock import patch
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import train_old_classifier_pretrain as launcher
from test_h4_output_scaling import runtime_guard
from RL.DGPO_neutrino.omnifold_ztautau.stage import build_fit_config
from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import EvenetAdapterRatioClassifier,pack_event_inputs
from RL.DGPO_neutrino.omnifold_ztautau.test_ztautau_omnifold import _FakeZtautauBackbone,_event_batch_with_pair_context
from diagnose_h4_classifier_calibration import resolve_classifier_settings


def test_weak_successful_classifier_contract():
    cfg=launcher.validated_config();runtime_guard(cfg)
    f=cfg['dgpo']['adaptive_omnifold']['audit_fit']
    fit=build_fit_config(f,n_train=20000,n_validation=4000);fit.validate()
    assert cfg['platform']['number_of_workers']==16
    assert (fit.min_steps,fit.steps)==(1000,3000)
    assert fit.adapter_learning_rate is None and fit.decoder_learning_rate is None
    base,override,source=resolve_classifier_settings(cfg)
    assert str(source)==launcher.SOURCE
    settings={**base,**override}
    for key in ('topology_pair_token','periodic_pair_features','train_last_pet_block','train_backbone'):
        assert settings[key] is False
    for key in ('train_grouped_sequential_embedding','train_invisible_projector'):
        assert settings[key] is True
    packed,spec=pack_event_inputs(_event_batch_with_pair_context(batch_size=4),include_pairwise_context=True)
    model=EvenetAdapterRatioClassifier(_FakeZtautauBackbone(),spec,decoder_hidden_dim=8,
        decoder_layers=1,decoder_heads=2,adapter_bottleneck=4,
        train_grouped_sequential_embedding=True,train_invisible_projector=True)
    assert not model._periodic_pair_features and model.bank.decoder.num_slots==2
    assert not hasattr(model.bank,'topology_encoder') and not hasattr(model.bank.decoder,'pair_in')
    # Metadata can be packed for diagnostics without becoming model features.
    assert model(packed,torch.randn(4,4)).shape==(4,)
    names={n for n,p in model.named_parameters() if p.requires_grad}
    assert any('GroupedSequentialEmbedding' in n for n in names)
    assert any('InvisibleInputProjector' in n for n in names)


@pytest.mark.parametrize('key,value',[('periodic_pair_features',True),('train_last_pet_block',True),('learning_rate',.01)])
def test_refuse_unintended_recipe_change(key,value):
    cfg=launcher.validated_config();cfg['dgpo']['adaptive_omnifold']['audit_fit'][key]=value
    with patch.object(launcher,'read_overlay_yaml',return_value=cfg):
        with pytest.raises(ValueError,match='recipe'):launcher.validated_config()


def test_ratio_export_preserves_pair_context_in_real_trainer():
    import ast
    t=ast.parse((launcher.ROOT/'evenet_dgpo/RL/DGPO_neutrino/dgpo_trainer.py').read_text())
    assignment=next(n for n in ast.walk(t) if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='classifier_pool' for x in n.targets))
    expr=next(k.value for k in assignment.value.keywords if k.arg=='include_pairwise_context')
    from types import SimpleNamespace
    cfg=SimpleNamespace(periodic_pair_features_enabled=False,audit_fit={'ratio_audit_export_dir':'export'})
    assert eval(compile(ast.Expression(expr),'<context-guard>','eval'),{'adaptive_cfg':cfg}) is True
    cfg.audit_fit={}
    assert eval(compile(ast.Expression(expr),'<context-guard>','eval'),{'adaptive_cfg':cfg}) is False
