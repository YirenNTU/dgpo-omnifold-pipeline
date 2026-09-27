import copy
from dataclasses import replace
from pathlib import Path
import sys
from unittest.mock import patch
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_h4_pair_token as launcher
from test_h4_output_scaling import runtime_guard
from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import (
    EvenetAdapterRatioClassifier, AdaLNZeroCandidateDecoder, pack_event_inputs,
    EvenetAdapterModelBuilder, peft_bank_factory)
from RL.DGPO_neutrino.omnifold_ztautau.test_ztautau_omnifold import (
    _FakeZtautauBackbone, _event_batch_with_pair_context)
from RL.DGPO_neutrino.omnifold_ztautau.stage import build_fit_config
from RL.DGPO_neutrino.omnifold_ztautau.ratio_fit import fit_density_ratio


def model_and_data():
    torch.set_num_threads(1)
    torch.manual_seed(31)
    packed, spec = pack_event_inputs(_event_batch_with_pair_context(batch_size=32),
                                     include_pairwise_context=True)
    backbone = _FakeZtautauBackbone()
    def build(spec):
        return EvenetAdapterRatioClassifier(backbone, spec,
            decoder_hidden_dim=8, decoder_layers=1, decoder_heads=2,
            adapter_bottleneck=4, periodic_pair_features=True,
            topology_max_harmonic=4, topology_pair_token=True, head_dropout=0.)
    return build(spec), packed, torch.randn(32, 4), build


def test_contract_and_actual_runtime_guard():
    cfg = launcher.validated_config()
    runtime_guard(cfg)
    fit = build_fit_config(cfg['dgpo']['adaptive_omnifold']['audit_fit'], n_train=20000, n_validation=4000)
    fit.validate()
    assert (fit.min_steps, fit.steps) == (1000, 3000)
    assert not fit.fourier_output_standardization
    assert cfg['dgpo']['adaptive_omnifold']['audit_fit']['disjoint_final_audit'] and fit.restore_best
    assert cfg['platform']['data_parquet_val_dir'] == launcher.CLEAN
    assert cfg['options']['Training']['model_checkpoint_load_path'] == launcher.SOURCE
    retry = launcher.with_suffix(cfg, 'retry1')
    runtime_guard(retry)
    assert retry['logger']['wandb']['id'] == 'h4pair01-retry1'
    assert retry['dgpo']['adaptive_omnifold']['audit_fit']['representation_export_dir'] is None
    assert 'h4_pair_token-retry1/' in retry['dgpo']['adaptive_omnifold']['audit_fit']['ratio_audit_export_dir']


@pytest.mark.parametrize('key,value', [('topology_pair_token',False),
    ('topology_fourier_embedding',True), ('fourier_output_standardization',True),
    ('learning_rate',.1)])
def test_reject_contract_drift(key,value):
    cfg=launcher.validated_config()
    cfg['dgpo']['adaptive_omnifold']['audit_fit'][key]=value
    with patch.object(launcher,'read_overlay_yaml',return_value=cfg):
        with pytest.raises(ValueError,match='Only pair-token'):
            launcher.validated_config()
    if key != 'learning_rate':
        with pytest.raises(ValueError,match='Pair-token fit'):
            runtime_guard(cfg)


def test_pair_token_participates_in_attention_and_has_no_bypass():
    m,c,z,_=model_and_data()
    assert not hasattr(m.bank,'topology_encoder')
    assert not hasattr(m.bank,'topology_standardizer')
    assert not hasattr(m.bank,'fusion')
    assert m.bank.decoder.num_slots==3 and m.bank.output.in_features==24
    seen=[]
    h=m.bank.decoder.blocks[0].self_attn.register_forward_pre_hook(lambda module,args:seen.append(args[0].shape))
    assert torch.equal(m(c,z),torch.zeros(32))
    h.remove()
    assert seen==[torch.Size([32,3,8])]
    # With gates open, pair features alter the physical-token streams too.
    d=m.bank.decoder
    for block in d.blocks:
        torch.nn.init.normal_(block.modulation.proj.bias, std=.2)
    args=dict(candidate_tokens=torch.randn(4,2,d.candidate_in.in_features),
        memory_tokens=torch.randn(4,3,d.memory_in.in_features),memory_mask=torch.ones(4,3,1,dtype=torch.bool),
        event_token=torch.randn(4,d.blocks[0].modulation.proj.in_features))
    a=d(**args,pair_features=torch.randn(4,9))
    b=d(**args,pair_features=torch.zeros(4,9))
    assert not torch.allclose(a[:,:2],b[:,:2])


def test_candidate_isolation_gradient_and_checkpoint_roundtrip():
    m,c,z,build=model_and_data()
    m.eval()
    torch.nn.init.normal_(m.bank.output.weight,std=.1)
    candidates=torch.stack((z,z+.1,z-.1),1)
    all_scores=m(c,candidates)
    separate=torch.stack([m(c,candidates[:,i]) for i in range(3)],1)
    assert torch.allclose(all_scores,separate,atol=1e-6)
    all_scores.sum().backward()
    g=m.bank.decoder.pair_in.weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum()>0
    payload=copy.deepcopy(m.peft_payload())
    assert payload['classifier_config']['topology_pair_token'] is True
    restored=EvenetAdapterRatioClassifier.from_peft_payload(payload,model_builder=build,device=torch.device('cpu'))
    restored.eval()
    assert torch.equal(all_scores,restored(c,candidates))
    assert not any('standardizer' in key for key in payload['state'])


def test_real_fit_with_pair_branch_diagnostics():
    cfg=launcher.validated_config()
    fit=build_fit_config(cfg['dgpo']['adaptive_omnifold']['audit_fit'],n_train=20000,n_validation=4000)
    m,c,z,_=model_and_data()
    data=(c,z,torch.ones(32),c,z+.2,torch.ones(32))
    before=m.bank.decoder.pair_in.weight.detach().clone()
    rows=[]
    result=fit_density_ratio(m,*data,replace(fit,steps=4,min_steps=0,batch_size=16,
        train_microbatch_size_per_rank=None,train_candidates_per_event=None,
        restore_best=False,representation_probe_rows=32,representation_probe_interval_steps=2,
        ratio_audit_export_dir=None,diagnostic_max_snapshots=0),42,validation=data,progress_callback=rows.append)
    assert result.steps_completed==4
    assert not torch.equal(before,m.bank.decoder.pair_in.weight)
    assert rows[0]['stability/representation_probe/initial/branches_complete']==1
    assert rows[0]['stability/representation_probe/initial/error']==0
    assert any('pair_token' in key for row in rows for key in row)


@pytest.mark.parametrize('extra', [dict(topology_fourier_embedding=True),
    dict(topology_conditioning=True),dict(topology_direct_logit=True),
    dict(periodic_pair_features=False),dict(conditional_residual_rank=2)])
def test_incompatible_branches_fail(extra):
    _,spec=pack_event_inputs(_event_batch_with_pair_context(batch_size=2),include_pairwise_context=True)
    kw=dict(periodic_pair_features=True,topology_pair_token=True,
        decoder_hidden_dim=8,decoder_layers=1,decoder_heads=2,adapter_bottleneck=4)
    kw.update(extra)
    with pytest.raises(ValueError):
        EvenetAdapterRatioClassifier(_FakeZtautauBackbone(),spec,**kw)


def test_builder_factory_and_replay_settings(tmp_path):
    from types import SimpleNamespace
    from RL.DGPO_neutrino.omnifold_ztautau import evenet_ratio as er
    from diagnose_h4_classifier_calibration import resolve_classifier_settings
    cfg=launcher.validated_config()
    base, overrides, _ = resolve_classifier_settings(cfg)
    settings = {**base, **overrides}
    assert settings['topology_pair_token'] is True
    assert settings['topology_fourier_embedding'] is False
    path=tmp_path/'source.ckpt';path.touch()
    with patch.object(er,'_config_with_pet_adapters',return_value=SimpleNamespace()), \
         patch('RL.DGPO_neutrino.model_utils.build_evenet_on_device',return_value=_FakeZtautauBackbone()), \
         patch('RL.DGPO_neutrino.model_utils.load_weights_like_configure_model'):
        builder=EvenetAdapterModelBuilder(config=SimpleNamespace(),normalization_dict={},
            checkpoint_path=path,device=torch.device('cpu'),
            periodic_pair_features=True,topology_pair_token=True,topology_max_harmonic=4,
            decoder_hidden_dim=8,decoder_layers=1,decoder_heads=2,head_dropout=0.)
    _,c,z,_=model_and_data()
    _,spec=pack_event_inputs(_event_batch_with_pair_context(batch_size=32),include_pairwise_context=True)
    with patch.object(builder,'_fresh_backbone',side_effect=_FakeZtautauBackbone):
        m=peft_bank_factory(builder,spec,classifier_overrides={'topology_pair_token':True})()
        assert m.bank.decoder.num_slots==3
        assert m(c,z).shape==(32,)
        legacy=builder.make_classifier(spec,topology_pair_token=False)
        assert legacy.bank.decoder.num_slots==2
        assert 'topology_pair_token' not in legacy.peft_payload()['classifier_config']


def test_bootstrap_handoff_includes_pair_token():
    # Execute the real trainer handoff without importing its GPU/Ray runtime.
    import ast
    source=(launcher.ROOT/'evenet_dgpo/RL/DGPO_neutrino/dgpo_trainer.py').read_text()
    tree=ast.parse(source)
    defaults=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='classifier_defaults' for t in n.targets))
    parent=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and defaults in n.body)
    start=parent.body.index(defaults)
    end=next(i for i in range(start,len(parent.body)) if isinstance(parent.body[i],ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='reward' for t in parent.body[i].targets))
    namespace={'recal':launcher.validated_config()['dgpo']['adaptive_omnifold']['recalibration'],
        '_dgpo_cfg_get':lambda c,k,d:c.get(k,d)}
    exec(compile(ast.Module(body=parent.body[start:end],type_ignores=[]),'<handoff>','exec'),namespace)
    assert namespace['classifier_config']['topology_pair_token'] is True
    assert namespace['classifier_config']['topology_fourier_embedding'] is False
