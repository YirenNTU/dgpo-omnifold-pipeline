"""Small local tests only; no remote training or W&B writes."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
import torch
from scripts.tau_explicit_inputs import (prepare_arrays,geometry_features,transform_panel_inputs,
                                        insert_geometry,build_explicit_classifier)
from scripts.train_conditional_spin_ratio import build_classifier,training_worker
from scripts.test_tau_conditioning import model_config
from scripts.run_tau_explicit_inputs import read_settings,make_config,add_comparison,ROOT
from scripts.test_tau_condition_width import synthetic_panel
from scripts.run_tau_bounded_ratio import bounded_panel_report


def fixture():
    rng=np.random.default_rng(42);n=48
    va=np.tile([12.,4.,2.,7.],(n,1));vb=np.tile([13.,-3.,1.,-8.],(n,1))
    va[:,0]+=np.arange(n)/20
    delta=rng.normal(0,.04,(n,2,2))
    gt=geometry_features(va,vb,delta)
    gg=geometry_features(va,vb,delta+.03)
    data=dict(condition=rng.normal(size=(n,7)).astype('float32'),
        candidate_truth=rng.normal(size=(n,29)).astype('float32'),
        candidate_generated=rng.normal(size=(n,29)).astype('float32'),
        event_weight=np.ones(n,dtype='float32'),split=np.r_[np.zeros(32),np.ones(8),np.full(8,2)],
        condition_mean=np.zeros(7),condition_scale=np.ones(7),
        category=np.tile([11,12,21,22],12),visible_pt_sum=np.arange(n,dtype=float),
        source_ids=np.arange(n).astype(str))
    return data,va,vb,gt,gg,delta


@pytest.mark.parametrize('arm',['visible','geometry','products'])
def test_initial_function_shared_weights_rng_and_input_gradients(arm):
    data,va,vb,gt,gg,_=fixture()
    out,spec=prepare_arrays(data,va,vb,gt,gg,arm)
    base=dict(model_config('film'),condition_hidden=256,condition_width=256,ratio_bound=30)
    cfg=dict(base,condition_dim=15,candidate_dim=out['candidate_truth'].shape[1],explicit_input=spec)
    torch.manual_seed(42);old=build_classifier(base);old_rng=torch.get_rng_state()
    torch.manual_seed(42);new=build_classifier(cfg);new_rng=torch.get_rng_state()
    assert torch.equal(old_rng,new_rng)
    for key,v in old.state_dict().items():
        w=new.state_dict()[key]
        if key=='condition_encoder.0.weight': w=w[:,:-8]
        if key=='spin_encoder.0.weight' and arm!='visible':
            extra=15 if arm=='products' else 6
            w=torch.cat((w[:,:-21-extra],w[:,-21:]),1)
        torch.testing.assert_close(v,w,rtol=0,atol=0)
    old.eval();new.eval()
    c=torch.from_numpy(out['condition']);t=torch.from_numpy(out['candidate_truth'])
    torch.testing.assert_close(old(torch.from_numpy(data['condition']),torch.from_numpy(data['candidate_truth'])),
                               new(c,t),rtol=1e-6,atol=1e-7)
    opt=torch.optim.AdamW(new.parameters(),lr=2e-4)
    for _ in range(3):
        opt.zero_grad();torch.nn.functional.softplus(new(c,t)).mean().backward();opt.step()
    assert new.condition_encoder[0].weight[:,-8:].norm()>0
    if arm=='geometry': assert new.spin_encoder[0].weight[:,-27:-21].norm()>0
    if arm=='products': assert new.spin_encoder[0].weight[:,-30:-21].norm()>0
    restored=build_classifier(cfg);restored.load_state_dict(new.state_dict(),strict=True);restored.eval()
    torch.testing.assert_close(new(c,t),restored(c,t),rtol=0,atol=0)


def test_fit_only_scaling_original_features_and_panel_parity():
    data,va,vb,gt,gg,delta=fixture()
    out,spec=prepare_arrays(data,va,vb,gt,gg,'geometry')
    changed=va.copy();changed[32:,0]+=10000
    _,spec2=prepare_arrays(data,changed,vb,gt,gg,'geometry')
    assert spec==spec2
    np.testing.assert_array_equal(out['condition'][:,:-8],data['condition'])
    np.testing.assert_array_equal(out['candidate_truth'][:,-21:],data['candidate_truth'][:,-21:])
    c,f=transform_panel_inputs(data['condition'],data['candidate_truth'],va,vb,delta,spec)
    np.testing.assert_array_equal(c,out['condition']);np.testing.assert_array_equal(f,out['candidate_truth'])
    # Swap candidate delta only: shared condition unchanged, geometry must change.
    c2,f2=transform_panel_inputs(data['condition'],data['candidate_truth'],va,vb,delta+.2,spec)
    np.testing.assert_array_equal(c,c2);assert not np.allclose(f,f2)
    # No kappa, truth identifier or Cij field enters the helper signature/spec.
    assert len(spec['geometry_fields'])==6 and len(spec['fields'])==8
    with pytest.raises(ValueError): insert_geometry(data['candidate_truth'],np.full((48,6),2.))


@pytest.mark.parametrize('arm',['visible','geometry','products'])
def test_actual_training_worker_saves_transform_and_can_reload(tmp_path,arm):
    import ray.train.torch
    data,va,vb,gt,gg,_=fixture()
    data,spec=prepare_arrays(data,va,vb,gt,gg,arm)
    np.savez(tmp_path/'prepared.npz',**data)
    cfg=dict(model_config('film'),explicit_input=spec,condition_width=256,condition_hidden=256,ratio_bound=30,
        seed=42,prepared=str(tmp_path/'prepared.npz'),checkpoint=str(tmp_path/'best.pt'),
        batch_size=8,backbone_cache='frozen',lr=2e-4,min_lr=1e-5,weight_decay=.001,
        epochs=2,min_delta=0.,min_steps=0,patience=3,representation='tau',packing_spec={},
        condition_normalization='masked_feature',ratio_objective='bce',mmd_coefficient=0.,
        skip_train_mmd_diagnostic=True,conditioning_diagnostics=True,diagnostic_every=1,
        condition_pt_edges=[10,20,30])
    reports=[]
    with patch('ray.train.get_context',return_value=SimpleNamespace(get_world_rank=lambda:0,get_world_size=lambda:1)), \
         patch('ray.train.torch.get_device',return_value=torch.device('cpu')), \
         patch('ray.train.torch.prepare_model',side_effect=lambda m:m),patch('ray.train.report',side_effect=reports.append), \
         patch('torch.distributed.all_reduce'),patch('torch.distributed.broadcast'):
        training_worker(cfg)
    saved=torch.load(cfg['checkpoint'],weights_only=True)
    assert saved['explicit_input']==spec and saved['condition_dim']==15
    model=build_classifier(saved);model.load_state_dict(saved['state_dict'],strict=True)
    assert 'explicit_visible_grad_norm' in reports[-1]
    if arm=='geometry': assert reports[-1]['explicit_geometry_weight_norm']>0
    if arm=='products':
        assert reports[-1]['explicit_products_weight_norm']>0
        assert reports[-1]['explicit_products_grad_norm']>0
        assert reports[-1]['explicit_geometry_weight_norm']>0
    assert all(np.isfinite(v) for v in reports[-1].values())


def test_config_and_paired_endpoint():
    settings=read_settings(ROOT/'config/conditional_tau_explicit_inputs_10pct.yaml')
    data,va,vb,gt,gg,_=fixture();out,spec=prepare_arrays(data,va,vb,gt,gg,'visible')
    settings.update(fresh_runtime='r',fresh_generator_checkpoint='raw1110',parameter_counts={},
        explicit_parameter_counts={},explicit_input=spec,input_inventory={},input_dimensions={'condition':15,'candidate':29})
    base=dict(model_config('film'),ratio_objective='bce',lr=2e-4,epochs=250)
    cfg=make_config(settings,base,Path('/base'),Path('/output'),'visible')
    assert settings['workers']==16 and settings['batch_size']==1024
    assert cfg['condition_dim']==15 and cfg['condition_width']==256 and cfg['ratio_bound']==30
    assert cfg['generated_samples']==cfg['policy_updates']==cfg['backbone_updates']==0
    inputs,data,control,prior=synthetic_panel()
    report=bounded_panel_report(inputs,data,control,20,42)
    add_comparison(report,inputs,data,control,20,42,'condition256',control,prior)
    np.testing.assert_allclose(report['comparisons']['bounded_minus_condition256']['error_change_ci95'],0)
    prior['arms']['bounded']['C'][0][0]+=1
    with pytest.raises(ValueError,match='failed replay'):
        add_comparison(report,inputs,data,control,20,42,'condition256',control,prior)


def test_all_sequential_and_prepare_never_trains(monkeypatch):
    from scripts import run_tau_explicit_inputs as launcher
    config=str(ROOT/'config/conditional_tau_explicit_inputs_10pct.yaml')
    with patch('sys.argv',['script',config,'all']),patch('subprocess.run') as call:
        launcher.main()
    assert [c.args[0][3] for c in call.call_args_list]==['visible','geometry']
    assert all(c.kwargs['check'] for c in call.call_args_list)
    monkeypatch.setattr(launcher,'prepare',lambda cfg:(None,None,None,None,None))
    with patch('sys.argv',['script',config,'prepare']),patch.object(launcher,'run_arm') as call:
        launcher.main()
    call.assert_not_called()


@pytest.mark.parametrize('arm',['visible','geometry','products'])
def test_production_panel_worker_replays_old_scores_and_transforms_new_inputs(tmp_path,monkeypatch,arm):
    import sys
    from scripts import run_tau_fresh_negatives as runner
    from scripts import tau_fresh_negatives as extractor
    data,va,vb,gt,gg,delta=fixture()
    new,spec=prepare_arrays(data,va,vb,gt,gg,arm)
    n=len(va);c=data['condition']
    # Synthetic frozen extractor retains candidate-specific information.
    def extract(policy,batch,d,va,vb,stats,micro):
        flat=d.cpu().numpy().reshape(len(d),4)
        return np.tile(flat,(1,8))[:,:29].astype('float32')
    old_cfg=dict(model_config('film'),condition_width=256,condition_hidden=256,ratio_bound=30)
    old=build_classifier(old_cfg).eval()
    new_cfg=dict(old_cfg,condition_dim=new['condition'].shape[1],candidate_dim=new['candidate_truth'].shape[1],explicit_input=spec)
    head=build_classifier(new_cfg).eval()
    # Make added coordinates active: this detects silently scoring the old path.
    with torch.no_grad():
        head.condition_encoder[0].weight[:,-8:].fill_(.04)
        for block in head.blocks: block.context.weight.fill_(.005)
        if arm=='geometry': head.spin_encoder[0].weight[:,-27:-21].fill_(.04)
        if arm=='products': head.spin_encoder[0].weight[:,-36:-21].fill_(.04)
    panel=tmp_path/'panel';panel.mkdir();base=tmp_path/'base';base.mkdir();output=tmp_path/'output';output.mkdir()
    ids=np.arange(n).astype(str)
    deltas=np.stack([delta+j*.001 for j in range(64)],1)
    old_logits=np.empty((n,64),np.float32)
    with torch.no_grad():
        for j in range(64):
            f=extract(None,None,torch.from_numpy(deltas[:,j]),None,None,None,None)
            old_logits[:,j]=old(torch.from_numpy(c),torch.from_numpy(f)).numpy()
    np.savez(panel/'inputs.npz',source_ids=ids,condition=c,raw_condition=c,visible_a=va,visible_b=vb)
    np.savez(panel/'samples_and_scores.npz',deltas=deltas,logits=old_logits)
    torch.save(dict(old_cfg,state_dict=old.state_dict()),base/'best.pt')
    torch.save(dict(new_cfg,state_dict=head.state_dict(),relative_preprocessing={},packing_spec={}),output/'best.pt')
    cfg=dict(panel_directory=str(panel),output=str(output),checkpoint=str(output/'best.pt'),
        baseline_directory=str(base),seed=42,batch_size=1024,feature_batch_size=256)
    from contextlib import nullcontext
    monkeypatch.setattr(extractor,'isolated_rng',lambda *a:nullcontext())
    monkeypatch.setattr(extractor,'load_policy',lambda *a:None)
    monkeypatch.setattr(extractor,'make_batch',lambda *a:None)
    monkeypatch.setattr(extractor,'candidate_features',extract)
    monkeypatch.setitem(sys.modules,'RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio',
        SimpleNamespace(EventPackingSpec=SimpleNamespace(from_dict=lambda x:x)))
    real_device=torch.device
    monkeypatch.setattr(runner.torch,'device',lambda d:real_device('cpu') if d=='cuda:0' else real_device(d))
    from unittest.mock import Mock
    np_proxy=Mock(wraps=np)
    np_proxy.arange=lambda rank,stop,step:np.arange(rank,n if stop==119002 else stop,step)
    np_proxy.float32=np.float32
    monkeypatch.setattr(runner,'np',np_proxy)
    runner.panel_worker(cfg,0)
    positions=np.arange(0,n,16)
    with np.load(output/'panel-00.npz') as f:
        assert f['logits'].shape==(len(positions),64)
        np.testing.assert_array_equal(f['source_ids'],ids[positions])
        cc,ff=transform_panel_inputs(c[positions],extract(None,None,torch.from_numpy(deltas[positions,63]),None,None,None,None),
                                    va[positions],vb[positions],deltas[positions,63],spec)
        with torch.no_grad(): expected=head(torch.from_numpy(cc),torch.from_numpy(ff)).numpy()
        np.testing.assert_allclose(f['logits'][:,63],expected,rtol=1e-6,atol=1e-6)
