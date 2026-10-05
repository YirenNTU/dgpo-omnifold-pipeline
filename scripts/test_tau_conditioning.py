"""Matched head, diagnostic and actual training-path tests; CPU only."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from scripts.train_conditional_spin_ratio import build_classifier, paired_loss, training_worker
from scripts.tau_conditioning_heads import ConditioningBlock, parameter_count
from scripts.tau_conditioning_diagnostics import fit_pt_edges, conditional_report
from scripts.run_tau_conditioning import read_settings, make_config, matched_control, main, ROOT


def model_config(kind):
    return dict(condition_dim=7,candidate_dim=29,hidden=16,dropout=0.,relative_dim=6,
                head_kind=kind,head_depth=3)


def test_exact_capacity_initial_function_and_gradients():
    torch.manual_seed(42); a=build_classifier(model_config('concat'))
    torch.manual_seed(42); b=build_classifier(model_config('film'))
    assert parameter_count(a)==parameter_count(b)
    for (ka,va),(kb,vb) in zip(a.state_dict().items(),b.state_dict().items()):
        assert ka==kb
        torch.testing.assert_close(va,vb,atol=0,rtol=0)
    c,t,g=torch.randn(9,7),torch.randn(9,29),torch.randn(9,29)
    torch.testing.assert_close(a(c,t),b(c,t),atol=0,rtol=0)
    for model in (a,b):
        loss,_,_=paired_loss(model,c,t,g,torch.ones(9));loss.backward()
        for block in model.blocks:
            assert block.context.weight.grad.norm()>0
        torch.optim.AdamW(model.parameters(),lr=1e-3).step()
        assert not torch.allclose(model(c,t),model(c.roll(1,0),t))
    assert not torch.allclose(a(c,t),b(c,t))


def test_concat_is_affine_concatenation_and_film_is_affine_modulation():
    h,c=torch.randn(5,4),torch.randn(5,4)
    for kind in ('concat','film'):
        block=ConditioningBlock(kind,width=4,dropout=0.)
        torch.nn.init.normal_(block.context.weight,std=.1)
        if kind=='concat':
            z=torch.nn.functional.linear(torch.cat((block.norm(h),c),-1),
                torch.cat((block.hidden.weight,block.context.weight),-1),block.hidden.bias)
        else:
            gamma,beta=block.context(c).chunk(2,-1)
            z=block.hidden((1+gamma)*block.norm(h)+beta)
        torch.testing.assert_close(block(h,c),block.output(torch.nn.functional.silu(z)))


def test_checkpoint_factory_keeps_legacy_and_roundtrips_new_heads():
    from scripts.train_conditional_spin_ratio import ConditionalSpinMLP
    cfg=model_config('concat');old=copy.deepcopy(cfg);del old['head_kind']
    assert isinstance(build_classifier(old),ConditionalSpinMLP)
    for kind in ('concat','film'):
        cfg=model_config(kind);m=build_classifier(cfg)
        restored=build_classifier(cfg);restored.load_state_dict(m.state_dict())
        c,t=torch.randn(3,7),torch.randn(3,29)
        torch.testing.assert_close(m(c,t),restored(c,t),atol=0,rtol=0)


def test_fit_only_edges_and_constant_weight_diagnostic():
    pt=np.arange(20,dtype=float);split=np.r_[np.zeros(10),np.ones(10)]
    edge=fit_pt_edges(pt,split);pt[10:]=1e8
    assert fit_pt_edges(pt,split)==edge
    x=np.random.default_rng(1).normal(size=(20,15));w=np.ones(20)
    cat=np.tile([11,22],10)
    d=conditional_report(x,x,w,np.zeros(20),cat,pt,edge)
    assert d['summary']['category_mass_tv']==pytest.approx(0,abs=1e-14)
    assert d['summary']['group_tau_error_reweighted']==pytest.approx(0,abs=1e-14)
    # A constant erroneous ratio cannot hide behind global normalization.
    d=conditional_report(x,x,w,np.full(20,np.log(2)),cat,pt,edge)
    assert d['summary']['category_mass_tv']==pytest.approx(0,abs=1e-14)
    assert d['summary']['group_log_mean_ratio_rms']==pytest.approx(np.log(2))
    q=np.where(cat==11,np.log(3),0.)
    d=conditional_report(x,x,w,q,cat,pt,edge)
    assert d['summary']['category_mass_tv']==pytest.approx(.25)


def test_protocol_changes_only_fusion_between_arms():
    settings=read_settings(ROOT/'config/conditional_tau_conditioning_10pct.yaml')
    settings.update(condition_pt_edges=[1,2,3],parameter_counts={'concat':100,'film':100})
    baseline=dict(seed=42,lr=2e-4,min_lr=1e-5,epochs=250,hidden=128,dropout=.05,
        weight_decay=.001,workers=16,batch_size=1024)
    a=make_config(settings,baseline,Path('/tmp/base'),Path('/tmp/c'),'concat')
    b=make_config(settings,baseline,Path('/tmp/base'),Path('/tmp/f'),'film')
    allowed={'head_kind','output','prepared','checkpoint','run_name','tags'}
    assert {k for k in a if a[k]!=b[k]}<=allowed
    assert a['ratio_objective']=='bce' and a['mmd_coefficient']==0
    assert a['min_steps']==1000 and a['patience']==25


def test_launcher_runs_two_new_heads_only():
    with patch('sys.argv',['run',str(ROOT/'config/conditional_tau_conditioning_10pct.yaml'),'all']), \
         patch('subprocess.run',return_value=SimpleNamespace(returncode=0)) as launch:
        main()
    assert [call.args[0][3] for call in launch.call_args_list]==['concat','film']


@pytest.mark.parametrize('kind',['concat','film'])
def test_actual_worker_diagnostics_checkpoint_and_reload(kind):
    import ray.train.torch
    rng=np.random.default_rng(42);n=48
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);split=np.r_[np.zeros(32),np.ones(16)]
        data=dict(condition=rng.normal(size=(n,7)).astype('float32'),
            candidate_truth=rng.normal(size=(n,29)).astype('float32'),
            candidate_generated=rng.normal(size=(n,29)).astype('float32'),
            event_weight=np.ones(n,dtype='float32'),split=split,
            condition_mean=np.zeros(7),condition_scale=np.ones(7),
            category=np.tile([11,12,21,22],12),visible_pt_sum=np.arange(n,dtype=float))
        np.savez(root/'data.npz',**data)
        cfg=dict(model_config(kind),seed=42,prepared=str(root/'data.npz'),checkpoint=str(root/'best.pt'),
            batch_size=8,backbone_cache='frozen',lr=2e-4,min_lr=1e-5,weight_decay=.001,
            epochs=2,min_delta=0.,min_steps=0,patience=3,representation='tau',packing_spec={},
            condition_normalization='masked_feature',ratio_objective='bce',mmd_coefficient=0.,
            skip_train_mmd_diagnostic=True,conditioning_diagnostics=True,diagnostic_every=1,
            condition_pt_edges=fit_pt_edges(data['visible_pt_sum'],split))
        reports=[]
        with patch('ray.train.get_context',return_value=SimpleNamespace(
                get_world_rank=lambda:0,get_world_size=lambda:1)), \
             patch('ray.train.torch.get_device',return_value=torch.device('cpu')), \
             patch('ray.train.torch.prepare_model',side_effect=lambda m:m), \
             patch('ray.train.report',side_effect=reports.append), \
             patch('torch.distributed.all_reduce'),patch('torch.distributed.broadcast'):
            training_worker(cfg)
        assert len(reports)==2 and reports[-1]['optimizer_steps']==8
        assert 'val_condition/category_mass_tv' in reports[-1]
        assert 'val_ratio_risk' in reports[-1]
        assert reports[-1]['condition_projection_weight_norm']>0
        assert all(np.isfinite(v) for v in reports[-1].values())
        saved=torch.load(cfg['checkpoint'],weights_only=True)
        assert saved['head_kind']==kind and saved['head_depth']==3
        model=build_classifier(saved);model.load_state_dict(saved['state_dict'])
        assert torch.isfinite(model(torch.from_numpy(data['condition']),
            torch.from_numpy(data['candidate_truth']))).all()


def test_matched_control_rejects_mismatched_identity():
    with tempfile.TemporaryDirectory() as d:
        root=Path(d);arm=root/'concat';arm.mkdir()
        (arm/'COMPLETE').touch()
        cfg=dict(baseline_directory='a',baseline_run='b',seed=42,workers=16,batch_size=1024,
            epochs=250,lr=.0002,min_lr=.00001,hidden=128,dropout=.05,weight_decay=.001,
            patience=25,min_delta=.0001,min_steps=1000,head_depth=3,condition_pt_edges=[1,2,3],
            ratio_objective='bce',parameter_counts={'concat':100,'film':100},head_kind='film')
        (arm/'manifest.json').write_text(json.dumps(dict(cfg,head_kind='concat')))
        (root/'concat_latest_completed.json').write_text(json.dumps({'output':str(arm)}))
        np.savez(arm/'test_scores.npz',source_ids=np.array(['x']),log_ratio=np.zeros(1))
        arrays=dict(source_ids=np.array(['x']),split=np.array([2]))
        assert 'matched_concat' in matched_control({'output_root':d},cfg,arrays)
        arrays['source_ids']=np.array(['y'])
        with pytest.raises(ValueError,match='identities'):
            matched_control({'output_root':d},cfg,arrays)


def test_endpoint_reports_use_saved_scores_and_same_cij_estimator():
    from scripts.run_tau_conditioning import finish_conditioning
    rng=np.random.default_rng(4);n=64
    def unit():
        x=rng.normal(size=(n,3));return x/np.linalg.norm(x,axis=1)[:,None]
    a=dict(split=np.full(n,2),source_ids=np.arange(n).astype(str),
        event_weight=np.ones(n),kappas=np.tile([1.,-.33],(n,1)),
        truth_a=unit(),truth_b=unit(),sample_a=unit(),sample_b=unit(),
        tau_truth=rng.normal(size=(n,15)),tau_generated=rng.normal(size=(n,15)),
        category=np.tile([11,12,21,22],16),visible_pt_sum=np.arange(n),base_log_ratio=np.zeros(n))
    p,q=rng.normal(size=n),rng.normal(size=n)
    logs=[]
    run=SimpleNamespace(summary={},save=lambda *args,**kwargs:None,log=logs.append)
    with tempfile.TemporaryDirectory() as d:
        cfg=dict(output=d,ratio_objective='bce',nnukl_c=0.,bootstrap=20,seed=42,
            evaluation_status='test fixture',head_kind='concat',condition_pt_edges=[16,32,48])
        finish_conditioning(cfg,a,p,q,{'log_ratio':q},run,{'output_root':d})
        assert run.summary['cij/candidate_minus_bce']==pytest.approx(0,abs=1e-12)
        assert 'conditional/candidate_ratio/category_mass_tv' in run.summary
        assert not run.summary['matched_concat_available']
        report=json.loads((Path(d)/'conditional_ratio_report.json').read_text())
        assert report['candidate_ratio']['pt_edges']==[16,32,48]
        with np.load(Path(d)/'test_scores.npz') as saved:
            np.testing.assert_equal(saved['log_ratio'],q)


def distributed_worker(rank, rendezvous, directory):
    from torch import distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2)
    try:
        torch.manual_seed(99)
        c,t,g=torch.randn(12,7),torch.randn(12,29),torch.randn(12,29)
        take=slice(rank*6,(rank+1)*6)
        for kind in ('concat','film'):
            torch.manual_seed(42);model=DDP(build_classifier(model_config(kind)))
            loss,_,_=paired_loss(model,c[take],t[take],g[take],torch.ones(6))
            loss.backward()
            torch.save({k:p.grad for k,p in model.module.named_parameters()},Path(directory)/f'{kind}-{rank}.pt')
    finally:
        dist.destroy_process_group()


def test_two_rank_gradients_equal_global_batch_for_both_heads():
    import torch.multiprocessing as mp
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(distributed_worker,args=(str(Path(directory)/'rendezvous'),directory),nprocs=2,join=True)
        torch.manual_seed(99)
        c,t,g=torch.randn(12,7),torch.randn(12,29),torch.randn(12,29)
        for kind in ('concat','film'):
            torch.manual_seed(42);model=build_classifier(model_config(kind))
            loss,_,_=paired_loss(model,c,t,g,torch.ones(12));loss.backward()
            for rank in range(2):
                actual=torch.load(Path(directory)/f'{kind}-{rank}.pt',weights_only=True)
                for key,param in model.named_parameters():
                    torch.testing.assert_close(actual[key],param.grad,atol=2e-7,rtol=2e-5)
