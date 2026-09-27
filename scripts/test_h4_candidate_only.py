import copy
from dataclasses import replace
from pathlib import Path
import sys
from unittest.mock import patch
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_h4_candidate_only as launcher
from test_h4_output_scaling import runtime_guard
from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import (
    EvenetAdapterRatioClassifier, pack_event_inputs)
from RL.DGPO_neutrino.omnifold_ztautau.test_ztautau_omnifold import (
    _FakeZtautauBackbone, _event_batch_with_pair_context)
from RL.DGPO_neutrino.omnifold_ztautau.stage import build_fit_config
from RL.DGPO_neutrino.omnifold_ztautau.ratio_fit import fit_density_ratio


def model_data():
    torch.set_num_threads(1); torch.manual_seed(31)
    c,spec=pack_event_inputs(_event_batch_with_pair_context(batch_size=32),include_pairwise_context=True)
    from diagnose_h4_classifier_calibration import resolve_classifier_settings
    base,overrides,_=resolve_classifier_settings(launcher.validated_config())
    settings={**base,**overrides}
    settings.pop('body_only_checkpoint')
    settings.update(decoder_hidden_dim=8,decoder_heads=2,adapter_bottleneck=4,
        train_last_pet_block=False,head_dropout=0.)
    backbone=_FakeZtautauBackbone()
    def build(spec):return EvenetAdapterRatioClassifier(backbone,spec,**settings)
    return build(spec),c,torch.randn(32,4),build


def test_contract_and_runtime():
    cfg=launcher.validated_config(); runtime_guard(cfg)
    assert cfg['platform']['number_of_workers']==16
    assert cfg['platform']['resources_per_worker']['GPU']==1
    assert cfg['options']['Training']['model_checkpoint_load_path']==launcher.SOURCE
    assert cfg['reward_config']['omnifold']['backbone_checkpoint']==launcher.PRETRAIN
    assert cfg['platform']['data_parquet_val_dir']==launcher.CLEAN
    fit=build_fit_config(cfg['dgpo']['adaptive_omnifold']['audit_fit'],n_train=20000,n_validation=4000)
    fit.validate(); assert (fit.min_steps,fit.steps)==(1000,3000) and fit.restore_best
    assert cfg['dgpo']['adaptive_omnifold']['audit_fit']['disjoint_final_audit']
    retry=launcher.with_suffix(cfg,'retry1');runtime_guard(retry)
    assert retry['logger']['wandb']['id']=='h4cand01-retry1'
    assert 'h4_candidate_only-retry1/' in retry['dgpo']['adaptive_omnifold']['audit_fit']['ratio_audit_export_dir']


@pytest.mark.parametrize('key,value',[('periodic_pair_features',True),('relation_token_count',4),
    ('topology_pair_token',True),('decoder_layers',2),('visible_pair_rest_frame',True)])
def test_reject_drift(key,value):
    cfg=launcher.validated_config();cfg['dgpo']['adaptive_omnifold']['audit_fit'][key]=value
    with patch.object(launcher,'read_overlay_yaml',return_value=cfg):
        with pytest.raises(ValueError):launcher.validated_config()
    with pytest.raises(ValueError,match='Candidate-only'):runtime_guard(cfg)


def test_direct_candidate_path_and_checkpoint():
    m,c,z,build=model_data();m.eval();d=m.bank.decoder
    assert d.num_slots==2 and len(d.blocks)==1 and m.bank.output.in_features==16
    assert not hasattr(d,'relation_tokens') and not hasattr(d,'pair_in')
    assert not hasattr(m.bank,'topology_encoder')
    assert torch.equal(m(c,z),torch.zeros(32))
    # Balanced labels on identical conditions must still produce a head gradient.
    loss=(torch.nn.functional.softplus(-m(c,z)).mean()+torch.nn.functional.softplus(m(c,z+.4)).mean())/2
    loss.backward();assert m.bank.output.weight.grad.abs().sum()>1e-8
    torch.nn.init.normal_(m.bank.output.weight,std=.1)
    candidates=torch.stack((z,z+.1),1)
    a=m(c,candidates)
    assert not torch.allclose(a[:,0],a[:,1])
    assert torch.allclose(a,torch.stack([m(c,x) for x in candidates.unbind(1)],1),atol=1e-6)
    payload=copy.deepcopy(m.peft_payload())
    restored=EvenetAdapterRatioClassifier.from_peft_payload(payload,model_builder=build,device=torch.device('cpu'))
    restored.eval();assert torch.allclose(a,restored(c,candidates),atol=1e-6,rtol=1e-6)


def test_short_fit_updates_candidate_encoder_and_keeps_diagnostics():
    m,c,z,_=model_data()
    fit=build_fit_config(launcher.validated_config()['dgpo']['adaptive_omnifold']['audit_fit'],n_train=20000,n_validation=4000)
    data=(c,z,torch.ones(32),c,z+.4,torch.ones(32));rows=[]
    before=m.bank.decoder.candidate_in.weight.detach().clone()
    out=fit_density_ratio(m,*data,replace(fit,steps=5,min_steps=0,batch_size=16,
        train_microbatch_size_per_rank=None,train_candidates_per_event=None,restore_best=False,
        representation_probe_rows=32,representation_probe_interval_steps=2,
        ratio_audit_export_dir=None,diagnostic_max_snapshots=0),42,validation=data,progress_callback=rows.append)
    assert out.steps_completed==5
    assert not torch.equal(before,m.bank.decoder.candidate_in.weight)
    assert rows[0]['stability/representation_probe/initial/error']==0
    assert rows[0]['stability/representation_probe/initial/branches_complete']==1


@pytest.mark.parametrize('gpus',[0,8,16])
def test_existing_cluster_gpu_guard(gpus):
    from unittest.mock import Mock
    ray=Mock();ray.is_initialized.side_effect=[False,True]
    ray.cluster_resources.return_value={'GPU':gpus}
    with patch.dict(sys.modules,{'ray':ray}),patch.dict('os.environ',{'RAY_ADDRESS':'auto'}):
        if gpus<16:
            with pytest.raises(RuntimeError,match='Requires 16 GPUs'):launcher.check_gpu_cluster()
        else:launcher.check_gpu_cluster()
    ray.init.assert_called_once_with(address='auto',logging_level='ERROR')
    ray.shutdown.assert_called_once()


def test_check_only_does_not_connect_or_launch(tmp_path):
    with patch.object(sys,'argv',['launcher','--check-only']),patch.object(launcher,'preflight',return_value=tmp_path), \
        patch.object(launcher,'check_gpu_cluster') as cluster,patch.object(launcher.subprocess,'run') as run:
        assert launcher.main()==0
    cluster.assert_not_called();run.assert_not_called()
