"""CPU regression tests for the sole condition-branch intervention."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.train_conditional_spin_ratio import build_classifier
from scripts.test_tau_conditioning import model_config
from scripts.run_tau_condition_width import read_settings,make_config,width_panel_report,ROOT
from scripts.run_tau_bounded_ratio import bounded_panel_report
from scripts.test_tau_bounded_ratio import MappingOnlySummary


def test_wide_condition_only_and_identical_start(tmp_path):
    cfg=dict(model_config('film'),ratio_bound=30)
    torch.manual_seed(42);old=build_classifier(cfg)
    torch.manual_seed(42);wide=build_classifier(dict(cfg,condition_hidden=256,condition_width=256))
    for key,value in old.state_dict().items():
        if key.startswith('condition_encoder.') or '.context.' in key: continue
        torch.testing.assert_close(value,wide.state_dict()[key],rtol=0,atol=0)
    assert wide.condition_encoder[0].out_features==256
    assert wide.condition_encoder[2].out_features==256
    assert all(block.context.weight.shape==(128,256) for block in wide.blocks)
    assert wide.spin_encoder[0].out_features==64 and wide.readout.in_features==64
    assert len(wide.blocks)==3
    old.eval();wide.eval()
    c,t=torch.randn(8,7),torch.randn(8,29)
    torch.testing.assert_close(old(c,t),wide(c,t),rtol=0,atol=0)
    # Zero context projections learn first, then the widened encoder receives gradients.
    opt=torch.optim.AdamW(wide.parameters(),lr=2e-4)
    for step in range(2):
        opt.zero_grad();loss=torch.nn.functional.softplus(wide(c,t)).mean();loss.backward()
        assert all(block.context.weight.grad.norm()>0 for block in wide.blocks)
        if step==1: assert wide.condition_encoder[0].weight.grad.norm()>0
        opt.step()
    payload=dict(cfg,condition_hidden=256,condition_width=256,state_dict=wide.state_dict())
    torch.save(payload,tmp_path/'wide.pt')
    saved=torch.load(tmp_path/'wide.pt',weights_only=True)
    restored=build_classifier(saved);restored.load_state_dict(saved['state_dict'],strict=True);restored.eval()
    torch.testing.assert_close(wide(c,t),restored(c,t),rtol=0,atol=0)
    # Explicit defaults must remain identical to old checkpoints without new keys.
    torch.manual_seed(42);explicit=build_classifier(dict(cfg,condition_hidden=cfg['hidden'],condition_width=64))
    for key,value in old.state_dict().items():
        torch.testing.assert_close(value,explicit.state_dict()[key],rtol=0,atol=0)


def test_config_preserves_protocol_and_uses_saved_control(tmp_path):
    settings=read_settings(ROOT/'config/conditional_tau_condition_width_10pct.yaml')
    settings.update(fresh_runtime='runtime',fresh_generator_checkpoint='raw1110',parameter_counts={'64':1,'256':2})
    base=dict(seed=42,lr=2e-4,min_lr=1e-5,epochs=250,hidden=128,dropout=.05,
        weight_decay=.001,workers=16,batch_size=1024,head_kind='film',head_depth=3,
        patience=25,min_delta=1e-4,min_steps=1000,relative_dim=6,ratio_objective='bce',mmd_coefficient=0.)
    cfg=make_config(settings,base,Path('/old'),tmp_path,'condition256')
    assert all(cfg[k]==v for k,v in base.items())
    assert cfg['condition_hidden']==cfg['condition_width']==256
    assert cfg['ratio_bound']==30 and cfg['fresh_negatives'] is False
    assert cfg['control_run']=='a7mczoed' and cfg['baseline_run']=='pzq0nl1i'
    assert cfg['policy_updates']==cfg['backbone_updates']==cfg['generated_samples']==0
    assert 'bounded_minus' not in cfg['run_name']


def synthetic_panel():
    rng=np.random.default_rng(4);n=32;k=64
    inputs=dict(source_ids=np.arange(n).astype(str),weight=np.ones(n),truth_cij=rng.normal(size=(n,9)),
                category=np.full(n,11),visible_pt_sum=np.arange(n))
    data=dict(logits=rng.normal(size=(n,k)),cij=rng.normal(size=(n,k,9)))
    control=np.minimum(data['logits']*.8,np.log(30))
    prior=bounded_panel_report(inputs,data,control,20,42)
    return inputs,data,control,prior


def test_paired_width_endpoint_and_control_replay():
    inputs,data,control,prior=synthetic_panel()
    report=width_panel_report(inputs,data,control,20,42,control,prior)
    assert report['primary']=='bounded_minus_condition64'
    assert report['control_endpoints_verified']
    comp=report['comparisons']['bounded_minus_condition64']
    assert comp['error_change']==pytest.approx(0)
    np.testing.assert_allclose(comp['error_change_ci95'],0)
    assert np.asarray(comp['absolute_component_error_change_ci95']).shape==(2,9)
    assert len(report['arms'])==5
    broken=copy.deepcopy(prior);broken['arms']['bounded']['C'][0][0]+=1
    with pytest.raises(ValueError,match='did not reproduce'):
        width_panel_report(inputs,data,control,20,42,control,broken)


def test_report_wires_saved_control_into_k1_k64_and_groups(tmp_path,monkeypatch):
    from scripts import run_tau_condition_width as launcher
    from scripts.run_tau_bounded_ratio import group_report
    inputs,data,control,prior=synthetic_panel()
    panel=tmp_path/'panel';panel.mkdir();output=tmp_path/'output';output.mkdir();old=tmp_path/'old';old.mkdir()
    np.savez(panel/'inputs.npz',**inputs);np.savez(panel/'samples_and_scores.npz',**data)
    np.savez(old/'test_scores.npz',source_ids=inputs['source_ids'],log_ratio=control[:,0])
    np.savez(old/'fixed_panel_scores.npz',source_ids=inputs['source_ids'],logits=control)
    (old/'fixed_K64_report.json').write_text(json.dumps(prior))
    calls=[];logs=[];saved=[]
    def fake_finish(cfg,arrays,p,q,scores,run,settings,*,extra_score_arms,panel_reporter):
        calls.append(extra_score_arms)
        report=panel_reporter(inputs,data,control,20,42)
        (output/'fixed_K64_report.json').write_text(json.dumps(report))
        groups=group_report(inputs,data,control,[8,16,24])
        (output/'fixed_K64_groups.json').write_text(json.dumps(groups))
    monkeypatch.setattr(launcher.bounded,'finish',fake_finish)
    monkeypatch.setitem(sys.modules,'wandb',SimpleNamespace(Table=lambda **kw:kw))
    run=SimpleNamespace(summary=MappingOnlySummary(),save=lambda *a,**kw:saved.append(a),log=logs.append)
    cfg=dict(output=str(output),panel_directory=str(panel),condition_pt_edges=[8,16,24])
    arrays=dict(split=np.full(32,2),source_ids=inputs['source_ids'])
    launcher.finish(cfg,arrays,np.zeros(32),np.zeros(32),{},run,dict(control_directory=str(old),control_run='a7mczoed'))
    np.testing.assert_array_equal(calls[0]['condition64'],control[:,0])
    assert run.summary['control_refits']==0 and run.summary['control_endpoints_verified']
    assert run.summary['K64/bounded_minus_condition64']==pytest.approx(0)
    assert len(logs[0]['K64/Cij']['data'])==45
    assert len(logs[0]['K64/condition_groups']['data'])==20
    assert len(saved)==2


def test_cli_prepare_never_trains(monkeypatch):
    from scripts import run_tau_condition_width as launcher
    calls=[]
    monkeypatch.setattr(launcher,'prepare',lambda cfg:(calls.append(cfg) or (None,None,None,None)))
    def fail(*a,**kw): raise AssertionError('prepare must not train')
    monkeypatch.setattr(launcher,'run_arm',fail)
    monkeypatch.setattr(sys,'argv',['script',str(ROOT/'config/conditional_tau_condition_width_10pct.yaml'),'prepare'])
    launcher.main()
    assert len(calls)==1


def test_preflight_accepts_logging_failed_control_and_rejects_changed_protocol(tmp_path):
    from scripts import run_tau_condition_width as launcher
    from scripts import run_tau_bounded_ratio as bounded
    from scripts.tau_cij_components import analyze
    settings=read_settings(ROOT/'config/conditional_tau_condition_width_10pct.yaml')
    source=tmp_path/'source';source.mkdir();directory=tmp_path/'control';directory.mkdir();panel=tmp_path/'panel';panel.mkdir()
    settings.update(baseline_directory=str(source),control_directory=str(directory),panel_directory=str(panel),
                    fresh_runtime='runtime',fresh_generator_checkpoint='raw1110')
    base=dict(model_config('film'),hidden=128,ratio_objective='bce',seed=42,workers=16,batch_size=1024)
    cfg=bounded.make_config(settings,base,source,directory,'bounded')
    (directory/'manifest.json').write_text(json.dumps(cfg));(directory/'wandb.json').write_text(json.dumps(dict(id='a7mczoed')))
    n=119002;ids=np.arange(n).astype(str)
    arrays=dict(condition=np.zeros((n,7),np.float32),candidate_truth=np.zeros((n,29),np.float32),
                split=np.full(n,2),source_ids=ids)
    np.savez(source/'prepared.npz',**arrays)
    (directory/'prepared.npz').symlink_to(source/'prepared.npz')
    narrow=build_classifier(dict(base,ratio_bound=30))
    torch.save(dict(base,ratio_bound=30,state_dict=narrow.state_dict()),directory/'best.pt')
    np.savez(directory/'test_scores.npz',source_ids=ids,log_ratio=np.zeros(n),generated_logits=np.zeros(n))
    np.savez(directory/'fixed_panel_scores.npz',source_ids=ids,logits=np.zeros((n,64),np.float32))
    inputs,data,control,prior=synthetic_panel()
    replay,_=analyze(inputs,data,dict(cap=30,bootstrap=20,bootstrap_seed=42,top_count=2))
    (panel/'cap_confirmation_report.json').write_text(json.dumps(replay))
    prior['events']=n
    (directory/'fixed_K64_report.json').write_text(json.dumps(prior))
    actual,scores=launcher.load_control(settings,base,arrays)
    assert actual['ratio_bound']==30 and scores.shape==(n,)
    assert not (directory/'COMPLETE').exists()
    cfg['batch_size']=256
    (directory/'manifest.json').write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='batch_size'):
        launcher.load_control(settings,base,arrays)
