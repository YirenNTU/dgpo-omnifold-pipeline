import copy
from dataclasses import replace
from pathlib import Path
import sys
from unittest.mock import patch
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_h4_relation_tokens as launcher
from test_h4_output_scaling import runtime_guard
from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import (
    EvenetAdapterRatioClassifier, EvenetAdapterModelBuilder, peft_bank_factory, pack_event_inputs)
from RL.DGPO_neutrino.omnifold_ztautau.test_ztautau_omnifold import (
    _FakeZtautauBackbone, _event_batch_with_pair_context)
from RL.DGPO_neutrino.omnifold_ztautau.stage import build_fit_config
from RL.DGPO_neutrino.omnifold_ztautau.ratio_fit import fit_density_ratio


def setup_model():
    torch.set_num_threads(1)
    torch.manual_seed(31)
    c, spec = pack_event_inputs(_event_batch_with_pair_context(batch_size=32), include_pairwise_context=True)
    backbone = _FakeZtautauBackbone()
    def build(spec):
        return EvenetAdapterRatioClassifier(backbone, spec, decoder_hidden_dim=8,
            decoder_layers=2, decoder_heads=2, adapter_bottleneck=4,
            relation_token_count=4, periodic_pair_features=False, head_dropout=0.)
    return build(spec), c, torch.randn(32,4), build


def test_contract():
    cfg=launcher.validated_config()
    runtime_guard(cfg)
    assert cfg['platform']['number_of_workers']==16
    assert cfg['options']['Training']['model_checkpoint_load_path']==launcher.SOURCE
    assert cfg['reward_config']['omnifold']['backbone_checkpoint']==launcher.PRETRAIN
    assert cfg['platform']['data_parquet_val_dir']==launcher.CLEAN
    fit=build_fit_config(cfg['dgpo']['adaptive_omnifold']['audit_fit'],n_train=20000,n_validation=4000)
    fit.validate()
    assert (fit.min_steps,fit.steps)==(1000,3000)
    assert fit.restore_best and not fit.fourier_output_standardization
    runtime_guard(launcher.with_suffix(cfg,'retry1'))


def test_whole_event_attention_and_padding():
    m,c,z,_=setup_model();m.eval();d=m.bank.decoder
    assert d.num_slots==1 and m.bank.output.in_features==8
    assert d.relation_tokens.shape==(4,8)
    assert not hasattr(d,'pair_in') and not hasattr(m.bank,'topology_encoder')
    assert torch.isfinite(m(c,z)).all()
    for block in d.blocks:
        torch.nn.init.normal_(block.modulation.proj.bias,std=.2)
    args=dict(candidate_tokens=torch.randn(4,2,d.candidate_in.in_features),
        memory_tokens=torch.randn(4,3,d.memory_in.in_features),
        memory_mask=torch.tensor([[[1],[1],[0]]]*4,dtype=torch.bool),
        event_token=torch.randn(4,d.blocks[0].modulation.proj.in_features))
    seen=[]
    h=d.blocks[0].cross_attn.register_forward_pre_hook(lambda mod, a:seen.append((a[0].shape,a[1].shape)))
    a=d(**args);h.remove()
    assert seen==[(torch.Size([4,4,8]),torch.Size([4,5,8]))]
    changed={**args,'candidate_tokens':args['candidate_tokens']+.3}
    assert not torch.allclose(a,d(**changed))
    mem=args['memory_tokens'].clone();mem[:,0]+=.3
    assert not torch.allclose(a,d(**{**args,'memory_tokens':mem}))
    mem=args['memory_tokens'].clone();mem[:,2]+=100
    assert torch.allclose(a,d(**{**args,'memory_tokens':mem}),atol=1e-6)
    # Pooling is independent of the ordering of the learned relation latents.
    with torch.no_grad():d.relation_tokens.copy_(d.relation_tokens.flip(0))
    assert torch.allclose(a,d(**args),atol=1e-6)


def test_roundtrip_candidate_isolation_and_gradients():
    m,c,z,build=setup_model();m.eval()
    torch.nn.init.normal_(m.bank.output.weight,std=.1)
    for block in m.bank.decoder.blocks:torch.nn.init.normal_(block.modulation.proj.bias,std=.2)
    candidates=torch.stack((z,z+.1,z-.1),1)
    a=m(c,candidates)
    assert torch.allclose(a,torch.stack([m(c,x) for x in candidates.unbind(1)],1),atol=1e-6)
    a.sum().backward()
    for parameter in (m.bank.decoder.relation_tokens,m.bank.decoder.candidate_in.weight,m.bank.decoder.memory_in.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum()>0
    payload=copy.deepcopy(m.peft_payload())
    assert payload['classifier_config']['relation_token_count']==4
    restored=EvenetAdapterRatioClassifier.from_peft_payload(payload,model_builder=build,device=torch.device('cpu'))
    restored.eval()
    assert all(torch.equal(v,restored.bank.state_dict()[k]) for k,v in payload['state'].items())
    assert torch.allclose(a,restored(c,candidates),atol=1e-6,rtol=1e-6)


def test_training_updates_and_diagnostics():
    m,c,z,_=setup_model()
    fit=build_fit_config(launcher.validated_config()['dgpo']['adaptive_omnifold']['audit_fit'],n_train=20000,n_validation=4000)
    data=(c,z,torch.ones(32),c,z+.2,torch.ones(32))
    before=m.bank.decoder.relation_tokens.detach().clone();rows=[]
    result=fit_density_ratio(m,*data,replace(fit,steps=5,min_steps=0,batch_size=16,
        train_microbatch_size_per_rank=None,train_candidates_per_event=None,restore_best=False,
        representation_probe_rows=32,representation_probe_interval_steps=2,
        ratio_audit_export_dir=None,diagnostic_max_snapshots=0),42,validation=data,progress_callback=rows.append)
    assert result.steps_completed==5
    assert not torch.equal(before,m.bank.decoder.relation_tokens)
    assert rows[0]['stability/representation_probe/initial/error']==0
    assert rows[0]['stability/representation_probe/initial/branches_complete']==1


@pytest.mark.parametrize('extra',[dict(periodic_pair_features=True),dict(visible_pair_rest_frame=True),dict(topology_pair_token=True),dict(topology_fourier_embedding=True)])
def test_reject_engineered_branches(extra):
    _,spec=pack_event_inputs(_event_batch_with_pair_context(batch_size=2))
    with pytest.raises(ValueError):
        EvenetAdapterRatioClassifier(_FakeZtautauBackbone(),spec,relation_token_count=4,
            decoder_hidden_dim=8,decoder_heads=2,**extra)


def test_builder_and_replay(tmp_path):
    from types import SimpleNamespace
    from RL.DGPO_neutrino.omnifold_ztautau import evenet_ratio as er
    from diagnose_h4_classifier_calibration import resolve_classifier_settings
    base,overrides,_=resolve_classifier_settings(launcher.validated_config())
    assert {**base,**overrides}['relation_token_count']==4
    path=tmp_path/'pretrain.ckpt';path.touch()
    with patch.object(er,'_config_with_pet_adapters',return_value=SimpleNamespace()),patch(
        'RL.DGPO_neutrino.model_utils.build_evenet_on_device',return_value=_FakeZtautauBackbone()),patch(
        'RL.DGPO_neutrino.model_utils.load_weights_like_configure_model'):
        builder=EvenetAdapterModelBuilder(config=SimpleNamespace(),normalization_dict={},checkpoint_path=path,
            device=torch.device('cpu'),relation_token_count=4,decoder_hidden_dim=8,decoder_layers=2,decoder_heads=2)
    _,c,z,_=setup_model();_,spec=pack_event_inputs(_event_batch_with_pair_context(batch_size=32),include_pairwise_context=True)
    with patch.object(builder,'_fresh_backbone',side_effect=_FakeZtautauBackbone):
        m=peft_bank_factory(builder,spec,classifier_overrides={'relation_token_count':4})()
        assert m.bank.decoder.relation_token_count==4 and m(c,z).shape==(32,)
        legacy=builder.make_classifier(spec,relation_token_count=0)
        assert legacy.bank.decoder.num_slots==2
        assert 'relation_token_count' not in legacy.peft_payload()['classifier_config']


def test_initial_balanced_bce_reaches_event_attention():
    m,c,z,_=setup_model();m.train()
    # Same event conditions for both labels: no event-only discrimination.
    loss=(torch.nn.functional.softplus(-m(c,z)).mean()+torch.nn.functional.softplus(m(c,z+.4)).mean())/2
    loss.backward()
    grads=[b.modulation.proj.weight.grad for b in m.bank.decoder.blocks]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum()>1e-8 for g in grads)
