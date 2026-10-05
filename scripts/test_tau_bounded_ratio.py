from pathlib import Path
import json
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from scripts.tau_bounded_ratio import bounded_log_ratio,bound_metrics
from scripts.train_conditional_spin_ratio import build_classifier,paired_loss,training_worker
from scripts.test_tau_conditioning import model_config
from scripts.run_tau_bounded_ratio import read_settings,make_config,bounded_panel_report,group_report,ROOT


class MappingOnlySummary(dict):
    """Match W&B SummaryDict.update: unlike dict, keyword updates are invalid."""
    def update(self, mapping):
        super().update(mapping)


def test_stable_bound_monotonic_unit_origin_and_gradients():
    z=torch.tensor([-10000.,-10.,0.,2.,5.,10.,10000.],dtype=torch.float64,requires_grad=True)
    s=bounded_log_ratio(z,30)
    assert torch.isfinite(s).all() and torch.all(s<=np.log(30))
    assert s[2].item()==pytest.approx(0,abs=1e-14)
    assert torch.all(torch.diff(s)>0)
    s.sum().backward()
    torch.testing.assert_close(z.grad,1-torch.exp(s)/30)
    assert z.grad[4]>0  # No hard-clamp dead gradient beyond log30.
    for cap in (0,1,float('inf'),float('nan')):
        with pytest.raises(ValueError): bounded_log_ratio(z,cap)


def test_factory_identical_parameters_and_roundtrip(tmp_path):
    cfg=model_config('film');torch.manual_seed(42);old=build_classifier(cfg)
    torch.manual_seed(42);new=build_classifier(dict(cfg,ratio_bound=30))
    for k,v in old.state_dict().items(): torch.testing.assert_close(v,new.state_dict()[k],atol=0,rtol=0)
    c,t,g=torch.randn(5,7),torch.randn(5,29),torch.randn(5,29)
    torch.testing.assert_close(new(c,t),bounded_log_ratio(old(c,t),30))
    loss,p,q=paired_loss(new,c,t,g,torch.ones(5));loss.backward()
    assert all(block.context.weight.grad.norm()>0 for block in new.blocks)
    expected=(torch.nn.functional.softplus(-p)+torch.nn.functional.softplus(q)).mean()/2
    torch.testing.assert_close(loss,expected)
    payload=dict(cfg,ratio_bound=30,state_dict=new.state_dict())
    torch.save(payload,tmp_path/'head.pt')
    saved=torch.load(tmp_path/'head.pt',weights_only=True)
    restored=build_classifier(saved);restored.load_state_dict(saved['state_dict'])
    torch.testing.assert_close(new(c,t),restored(c,t),atol=0,rtol=0)
    with pytest.raises(ValueError): build_classifier(dict(cfg,ratio_bound=30,ratio_objective='mlc'))


def test_cap_population_bce_optimum():
    # Pointwise risk q*softplus(s)+p*softplus(-s); optimum min(p/q,30).
    s=torch.linspace(-8,np.log(30),10001,dtype=torch.float64)
    for ratio in (.1,1,10,100):
        risk=torch.nn.functional.softplus(s)+ratio*torch.nn.functional.softplus(-s)
        estimate=s[torch.argmin(risk)].item()
        assert estimate==pytest.approx(np.log(min(ratio,30)),abs=.002)


def test_manifest_no_training_changes_except_bound(tmp_path):
    settings=read_settings(ROOT/'config/conditional_tau_bounded_ratio_10pct.yaml')
    settings.update(fresh_runtime='runtime',fresh_generator_checkpoint='raw1110')
    base=dict(seed=42,lr=2e-4,min_lr=1e-5,epochs=250,hidden=128,dropout=.05,
        weight_decay=.001,workers=16,batch_size=1024,head_kind='film',head_depth=3,
        patience=25,min_delta=1e-4,min_steps=1000,relative_dim=6,ratio_objective='bce',mmd_coefficient=0.)
    cfg=make_config(settings,base,Path('/control'),tmp_path,'bounded')
    assert all(cfg[k]==v for k,v in base.items())
    assert cfg['ratio_bound']==30 and cfg['fresh_negatives'] is False
    assert cfg['generated_samples']==cfg['policy_updates']==cfg['backbone_updates']==0
    assert 'fresh_inputs' not in cfg
    with pytest.raises(ValueError): make_config(settings,dict(base,ratio_bound=30),Path('/c'),tmp_path,'bounded')


def test_K64_report_matched_control_and_groups():
    rng=np.random.default_rng(42);n,k=40,8
    inputs=dict(weight=np.ones(n),truth_cij=rng.normal(size=(n,9)),category=np.full(n,11),visible_pt_sum=np.arange(n))
    data=dict(logits=rng.normal(size=(n,k))*3,cij=rng.normal(size=(n,k,9)))
    scores=np.minimum(data['logits'],np.log(30))
    report=bounded_panel_report(inputs,data,scores,30,42)
    comp=report['comparisons']['bounded_minus_old_cap30']
    assert comp['error_change']==pytest.approx(0)
    np.testing.assert_allclose(comp['error_change_ci95'],0)
    assert 'fresh_raw' not in report['arms']
    assert np.asarray(comp['absolute_component_error_change_ci95']).shape==(2,9)
    groups=group_report(inputs,data,scores,[10,20,30])
    assert len(groups['groups'])==15
    a=[r for r in groups['groups'] if r['arm']=='bounded']
    b=[r for r in groups['groups'] if r['arm']=='old_cap30']
    for x,y in zip(a,b): np.testing.assert_allclose(x['C'],y['C'])
    with pytest.raises(ValueError): bounded_panel_report(inputs,data,np.full((n,k),4.),30,42)


@pytest.mark.parametrize('condition_width',[64,256])
def test_actual_training_saves_bound_and_logs_saturation(tmp_path,condition_width):
    import ray.train.torch
    rng=np.random.default_rng(42);n=48
    data=dict(condition=rng.normal(size=(n,7)).astype('float32'),
        candidate_truth=rng.normal(size=(n,29)).astype('float32'),
        candidate_generated=rng.normal(size=(n,29)).astype('float32'),
        event_weight=np.ones(n,dtype='float32'),split=np.r_[np.zeros(32),np.ones(16)],
        condition_mean=np.zeros(7),condition_scale=np.ones(7))
    np.savez(tmp_path/'prepared.npz',**data)
    cfg=dict(model_config('film'),seed=42,prepared=str(tmp_path/'prepared.npz'),
        checkpoint=str(tmp_path/'best.pt'),batch_size=8,lr=2e-4,min_lr=1e-5,weight_decay=.001,
        epochs=2,min_delta=0.,min_steps=0,patience=3,representation='tau',packing_spec={},
        condition_normalization='masked_feature',ratio_objective='bce',mmd_coefficient=0.,
        skip_train_mmd_diagnostic=True,ratio_bound=30,fresh_negatives=False,
        condition_width=condition_width,condition_hidden=128 if condition_width==64 else 256)
    reports=[]
    with patch('ray.train.get_context',return_value=SimpleNamespace(get_world_rank=lambda:0,get_world_size=lambda:1)), \
         patch('ray.train.torch.get_device',return_value=torch.device('cpu')), \
         patch('ray.train.torch.prepare_model',side_effect=lambda m:m), \
         patch('ray.train.report',side_effect=reports.append), \
         patch('torch.distributed.all_reduce'),patch('torch.distributed.broadcast'):
        training_worker(cfg)
    saved=torch.load(cfg['checkpoint'],weights_only=True)
    assert saved['ratio_bound']==30
    assert saved['condition_width']==condition_width
    m=build_classifier(saved);m.load_state_dict(saved['state_dict'])
    assert m.condition_encoder[2].out_features==condition_width
    assert reports[-1]['optimizer_steps']==8
    assert reports[-1]['val_bound/generated_ratio_max']<=30.00001
    assert 0<=reports[-1]['val_bound/generated_near_cap_fraction']<=1
    assert 0<=reports[-1]['val_bound/generated_mean_output_jacobian']<=1


def test_endpoint_orchestration_16_shards_and_source_replay(tmp_path,monkeypatch):
    from scripts.run_tau_bounded_ratio import finish
    from scripts.tau_cij_components import analyze
    rng=np.random.default_rng(4);n=32;k=64
    panel=tmp_path/'panel';panel.mkdir();output=tmp_path/'output';output.mkdir()
    inputs=dict(source_ids=np.arange(n).astype(str),weight=np.ones(n),
        truth_cij=rng.normal(size=(n,9)),category=np.full(n,11),visible_pt_sum=np.arange(n))
    data=dict(logits=rng.normal(size=(n,k))*2,cij=rng.normal(size=(n,k,9)))
    np.savez(panel/'inputs.npz',**inputs)
    np.savez(panel/'samples_and_scores.npz',source_ids=inputs['source_ids'],**data)
    prior,_=analyze(inputs,data,dict(cap=30,bootstrap=20,bootstrap_seed=42,top_count=2))
    (panel/'cap_confirmation_report.json').write_text(json.dumps(prior))
    cfg=dict(panel_directory=str(panel),output=str(output),bootstrap=20,seed=42,condition_pt_edges=[8,16,24])
    scores=np.minimum(data['logits'],np.log(30)).astype(np.float32)
    ranks=[]
    class Remote:
        def remote(self,cfg,rank):
            ranks.append(rank);idx=np.arange(rank,n,16)
            np.savez(output/f'panel-{rank:02d}.npz',positions=idx,source_ids=inputs['source_ids'][idx],logits=scores[idx])
            return rank
    def remote(**kwargs):
        assert kwargs['num_gpus']==1 and kwargs['max_calls']==1
        return lambda fn:Remote()
    monkeypatch.setitem(sys.modules,'ray',SimpleNamespace(remote=remote,get=lambda x:x))
    monkeypatch.setitem(sys.modules,'wandb',SimpleNamespace(Table=lambda **kw:kw))
    k1=[];logs=[]
    monkeypatch.setattr('scripts.run_tau_bounded_ratio.finish_reports',lambda *a,**kw:k1.append(kw))
    run=SimpleNamespace(summary=MappingOnlySummary(),save=lambda *a,**kw:None,log=logs.append)
    arrays=dict(event_weight=np.ones(n),split=np.full(n,2))
    finish(cfg,arrays,np.zeros(n),np.zeros(n),dict(log_ratio=np.zeros(n)),run,{})
    assert ranks==list(range(16))
    assert 'old_cap30' in k1[0]['extra_score_arms']
    assert run.summary['source_endpoints_verified']
    assert abs(run.summary['K64/bounded_minus_old_cap30'])<1e-7
    assert len(logs[0]['K64/Cij']['data'])==36
    report=json.loads((output/'fixed_K64_report.json').read_text())
    assert report['ratio_bound']==30


def test_report_recovery_without_training_or_inference(tmp_path,monkeypatch):
    from scripts import run_tau_bounded_ratio as launcher
    from scripts.tau_cij_components import analyze
    settings=read_settings(ROOT/'config/conditional_tau_bounded_ratio_10pct.yaml')
    panel=tmp_path/'panel';panel.mkdir();output=tmp_path/'bounded';output.mkdir()
    settings['panel_directory']=str(panel)
    cfg={k:settings[k] for k in ('baseline_run','panel_run','ratio_bound','workers','batch_size','panel_directory')}
    cfg.update(head_kind='film',ratio_objective='bce',fresh_negatives=False)
    rng=np.random.default_rng(9);n=32;k=64
    inputs=dict(source_ids=np.arange(n).astype(str),weight=np.ones(n),truth_cij=rng.normal(size=(n,9)),
                category=np.full(n,11),visible_pt_sum=np.arange(n))
    data=dict(logits=rng.normal(size=(n,k)),cij=rng.normal(size=(n,k,9)))
    logits=np.minimum(data['logits'],np.log(30))
    prior,_=analyze(inputs,data,dict(cap=30,bootstrap=20,bootstrap_seed=42,top_count=2))
    (panel/'cap_confirmation_report.json').write_text(json.dumps(prior))
    report=bounded_panel_report(inputs,data,logits,20,42)
    # Small synthetic endpoint fixture; exercise production report-shape metadata.
    report['events']=119002
    groups=group_report(inputs,data,logits,[8,16,24])
    for name,payload in [('manifest.json',cfg),('wandb.json',dict(id='a7mczoed')),
                         ('fixed_K64_report.json',report),('fixed_K64_groups.json',groups)]:
        (output/name).write_text(json.dumps(payload))
    before={p.name:p.read_bytes() for p in output.iterdir()}
    saved=[];logs=[];init_calls=[]
    class Run:
        id='new-report-only';url='https://example.invalid/new-report-only'
        summary=MappingOnlySummary()
        def __enter__(self): return self
        def __exit__(self,*exc): return False
        def save(self,path,**kw): saved.append(Path(path).name)
        def log(self,payload): logs.append(payload)
    run=Run()
    def init(**kwargs): init_calls.append(kwargs);return run
    monkeypatch.setitem(sys.modules,'wandb',SimpleNamespace(init=init,Table=lambda **kw:kw))
    monkeypatch.setitem(sys.modules,'ray',None)
    def forbidden(*a,**kw): raise AssertionError('Recovery must not train, prepare, score or load models')
    for name in ('prepare_control','run_arm','panel_worker','finish'):
        monkeypatch.setattr(launcher,name,forbidden)
    monkeypatch.setattr(torch,'load',forbidden)
    launcher.recover_report(settings,output)
    assert saved==['fixed_K64_report.json','fixed_K64_groups.json']
    assert run.summary['phase']=='complete' and run.summary['source_run']=='a7mczoed'
    assert run.summary['classifier_fits']==run.summary['inference_calls']==0
    assert run.summary['source_endpoints_verified']
    assert len(logs[0]['K64/Cij']['data'])==36
    assert len(logs[0]['K64/condition_groups']['data'])==15
    assert init_calls[0]['config']['source_run']=='a7mczoed'
    assert 'id' not in init_calls[0] and 'resume' not in init_calls[0]
    assert {name:(output/name).read_bytes() for name in before}==before
    assert not (output/'COMPLETE').exists()
    assert len(list(output.glob('report-recovery-*/COMPLETE')))==1
    # Verification must happen before a new W&B run is created.
    report['arms']['old_cap30']['error']+=1
    (output/'fixed_K64_report.json').write_text(json.dumps(report))
    with pytest.raises(ValueError,match='did not reproduce'):
        launcher.recover_report(settings,output)
    assert len(init_calls)==1


def test_report_cli_skips_training_prepare(tmp_path,monkeypatch):
    from scripts import run_tau_bounded_ratio as launcher
    calls=[]
    monkeypatch.setattr(launcher,'recover_report',lambda settings,directory:calls.append(directory))
    def forbidden(*a,**kw): raise AssertionError('No data preparation for report-only')
    monkeypatch.setattr(launcher,'prepare_control',forbidden)
    monkeypatch.setattr(sys,'argv',['run_tau_bounded_ratio.py',str(ROOT/'config/conditional_tau_bounded_ratio_10pct.yaml'),
                                    'report','--directory',str(tmp_path)])
    launcher.main()
    assert calls==[tmp_path]
