"""CPU checks for nested inference, raw ratios, and clustered uncertainty."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
import torch

from scripts.tau_sampling_convergence import aggregates, conditional_mc_se, convergence_report, group_report
from scripts.run_tau_sampling_convergence import read_settings, merge, physics_features, score_draw, prepare, plot_report, ROOT


def fixture(n=48,d=9):
    rng=np.random.default_rng(8)
    return (rng.normal(size=(n,d)),rng.normal(size=(n,8,d)),
            rng.normal(size=(n,8)),rng.uniform(.5,2,n))


def test_global_ratio_exact_formula_and_constant_shift():
    t,g,s,w=fixture()
    for k in (1,2,4,8):
        num,den,_,_=aggregates(g,s,w,k)
        direct=(w[:,None,None]*np.exp(s[:,:k,None])*g[:,:k]).sum((0,1))/(w[:,None]*np.exp(s[:,:k])).sum()
        np.testing.assert_allclose(num[:,1].sum(0)/den[:,1].sum(),direct)
        num2,den2,_,_=aggregates(g,s+1000,w,k)
        np.testing.assert_allclose(num2.sum(0)/den2.sum(0)[:,None],num.sum(0)/den.sum(0)[:,None],atol=1e-12)
    # Unequal event mean ratios MUST change event mass (no per-event normalization).
    num,den,_,_=aggregates(g,np.repeat(np.log(w)[:,None],8,1),w,8)
    np.testing.assert_allclose(den[:,1]/den[:,1].sum(),w*w/(w*w).sum())


def test_zero_base_weight_extreme_logit_has_no_effect():
    t,g,s,w=fixture();w[0]=0;s[0]=1e6
    num,den,r,_=aggregates(g,s,w,8)
    assert np.isfinite(num).all() and np.isfinite(den).all() and (r[0]==0).all()


def test_constant_ratio_identity_and_identical_candidate_clusters(tmp_path):
    t,g,s,w=fixture()
    g=np.repeat(t[:,None],8,axis=1);s[:]=2
    report=convergence_report(t,g,s,w,bootstrap=20)
    for k,row in report['prefixes'].items():
        np.testing.assert_allclose(row['reweighted'],report['truth'],atol=1e-14)
        np.testing.assert_allclose(row['error_change_ci95'],0,atol=1e-14)
        assert row['log_mean_ratio']==pytest.approx(2)
        if int(k)>1: np.testing.assert_allclose(row['conditional_mc_se'],0,atol=1e-14)
    assert report['prefixes']['1']['conditional_mc_se'] is None
    # Duplicating candidates does not fabricate a larger independent condition population.
    np.testing.assert_allclose(report['prefixes']['1']['matrix_ci95'],report['prefixes']['8']['matrix_ci95'],atol=1e-14)
    plot_report(report,tmp_path/'curve.png')
    assert (tmp_path/'curve.png').stat().st_size>1000


def test_nested_prefix_and_disjoint_panel_accounting():
    t,g,s,w=fixture()
    a=convergence_report(t,g,s,w,bootstrap=20,seed=19)
    g2=g.copy();g2[:,4:]+=30
    b=convergence_report(t,g2,s,w,bootstrap=20,seed=19)
    for k in ('1','2','4'): assert a['prefixes'][k]==b['prefixes'][k]
    assert [len(a['disjoint_panels'][str(k)]) for k in (1,2,4,8)]==[8,4,2,1]
    assert a['prefixes']['8']['reweighted']!=b['prefixes']['8']['reweighted']


def test_mc_se_matches_direct_variance_with_unit_ratio():
    t,g,s,w=fixture();s[:]=0
    num,den,r,_=aggregates(g,s,w,8)
    m=num.sum(0)/den.sum(0)[:,None]
    se=conditional_mc_se(g,r,w,m)
    expected=np.sqrt(((w/w.sum())[:,None]**2*g.var(axis=1,ddof=1)).sum(0)/8)
    np.testing.assert_allclose(se[0],expected);np.testing.assert_allclose(se[1],expected)
    np.testing.assert_allclose(se[2],0,atol=1e-15)


def test_group_k1_matches_existing_diagnostic():
    from scripts.tau_conditioning_diagnostics import conditional_report
    t,g,s,w=fixture(d=15);cat=np.tile([11,12,21,22],12);pt=np.arange(48)
    old=conditional_report(t,g[:,0],w,s[:,0],cat,pt,[12,24,36])
    new=group_report(t,g,s,w,cat,pt,[12,24,36],[1,2,4,8])
    for a,b in zip(old['groups'],new['1']):
        assert a['group']==b['group']
        for key in ('base_mass','weighted_mass','log_mean_ratio'):
            assert a[key]==pytest.approx(b[key])
        assert a['tau_error_reweighted']==pytest.approx(b['reweighted_error'])


def test_shard_merge_16_workers_identity_and_missing(tmp_path):
    n=35;ids=np.array([str(i) for i in range(n)])
    cfg=dict(output=str(tmp_path),workers=16,prefixes=[1,2,4,8])
    for rank in range(16):
        pos=np.arange(rank,n,16)
        for j in range(8):
            np.savez(tmp_path/f'rank-{rank:02d}-candidate-{j:02d}.npz',positions=pos,source_ids=ids[pos],
                deltas=np.full((len(pos),2,2),j,dtype='float32'),logits=pos+j,
                tau=np.ones((len(pos),15)),cij=np.ones((len(pos),9)),replay_max_error=.00001)
    a,error=merge(cfg,ids)
    assert a['logits'].shape==(n,8) and error==.00001
    np.testing.assert_equal(a['logits'],np.arange(n)[:,None]+np.arange(8))
    with pytest.raises(ValueError,match='identities'):merge(cfg,ids[::-1])


def test_scoring_uses_native_hidden_fit_scaling_and_eval_head():
    from scripts.train_conditional_spin_ratio import build_classifier
    cfg=dict(head_kind='film',head_depth=3,condition_dim=7,candidate_dim=25,hidden=16,dropout=.05,relative_dim=6,
             relative_preprocessing=dict(mean=[.1]*6,scale=[2.]*6))
    head=build_classifier(cfg).eval();rng=np.random.default_rng(2)
    c=rng.normal(size=(6,7)).astype('float32')
    va=np.tile([20.,5.,3.,18.],(6,1));vb=np.tile([18.,-7.,-3.,-15.],(6,1))
    delta=torch.full((6,2,2),.05);kap=np.ones((6,2))
    hidden=rng.normal(size=(6,4)).astype('float32')
    def fake_hidden(policy,batch,candidate):return hidden[batch['indices'].numpy()]
    with patch('scripts.tau_backbone_alignment.candidate_hidden',side_effect=fake_hidden):
        score,tau,cij=score_draw(None,head,cfg,{'indices':torch.arange(6)},delta,c,va,vb,kap,'cpu',2)
    _,rel,_=physics_features(delta.numpy(),va,vb,kap)
    features=np.concatenate((hidden,((rel-.1)/2).astype('float32'),tau),1)
    with torch.no_grad(): expected=head(torch.from_numpy(c),torch.from_numpy(features)).numpy()
    np.testing.assert_allclose(score,expected);assert cij.shape==(6,9)


def test_prepare_rebuilds_conditions_truth_and_historical_order(tmp_path):
    from scripts.conditional_tau_preprocessing import apply_masked_feature
    from scripts.diagnose_ztautau_cij import source_ids,angles,tau_from_deltas
    from scripts.train_conditional_spin_ratio import build_classifier
    n=12;source=tmp_path/'classifier';source.mkdir();events=tmp_path/'events'/'val';events.mkdir(parents=True)
    (events.parent/'filter_manifest.json').write_text('{"complete":true}')
    spec={'shapes':{'x':[2,3],'x_mask':[2]}}
    raw=np.tile([1,2,3,4,5,6,1,1],(n,1)).astype('float32')
    delta=np.full((n,2,2),.05,dtype='float32')
    pool=dict(condition=torch.tensor(raw),truth=torch.tensor(delta),source_sample_index=torch.ones(n,dtype=torch.int64),
              source_event_key=torch.arange(n),invalid_target_rows=0)
    ids=source_ids(pool)
    va=np.tile([20.,5.,3.,18.],(n,1));vb=np.tile([18.,-7.,-3.,-15.],(n,1));kap=np.ones((n,2))
    columns=dict(source_sample_index=np.ones(n,dtype=int),source_event_key=np.arange(n),event_weight=np.ones(n),
                 event_category=np.full(n,11),analyzing_power_a=kap[:,0],analyzing_power_b=kap[:,1])
    for leg,values in [('a',va),('b',vb)]:
        columns.update({f'lead_{leg}_visible_{key}':values[:,j] for j,key in enumerate(('E','px','py','pz'))})
    table={key:SimpleNamespace(to_numpy=lambda value=value:value) for key,value in columns.items()}
    model_cfg=dict(head_kind='film',head_depth=3,relative_dim=6,condition_dim=24,candidate_dim=25,hidden=16,dropout=0.)
    saved=dict(model_cfg,state_dict=build_classifier(model_cfg).state_dict(),condition_normalization='masked_feature',
               packing_spec=spec,condition_mean=torch.zeros(24),condition_scale=torch.ones(24),epoch=249)
    torch.save(saved,source/'best.pt')
    manifest=dict(model_cfg,ratio_objective='bce',test_events=str(events),packing_spec=spec,
        backbone_manifest=dict(global_step=1110,weights='raw_state_dict_only',checkpoint='/raw.ckpt',runtime='/runtime.yaml'),
        test_sample_manifest=dict(checkpoint='/raw.ckpt',weights='raw_state_dict_only',ddim_steps=20),
        test_source=str(source),condition_pt_edges=[1,2,3])
    (source/'manifest.json').write_text(json.dumps(manifest));(source/'wandb.json').write_text('{"id":"pinned"}')
    (source/'COMPLETE').write_text('yes')
    (source/'cij_comparison.json').write_text(json.dumps({'arms':{'unweighted':{'error':0},'candidate_ratio':{'error':0}}}))
    tau,_,_=physics_features(delta,va,vb,kap)
    a,b=angles(tau_from_deltas(va,delta[:,0]),tau_from_deltas(vb,delta[:,1]),va,vb)
    cond=apply_masked_feature(np.concatenate((raw,np.eye(16,dtype='float32')[np.zeros(n,int)]),1),np.zeros(24),np.ones(24),spec)
    # Stored order differs from parquet; all fields must follow identities.
    order=np.arange(n)[::-1]
    np.savez(source/'prepared.npz',split=np.full(n,2),source_ids=ids[order],condition=cond[order],tau_truth=tau[order],
             event_weight=np.ones(n),category=np.full(n,11),kappas=kap,truth_a=a[order],truth_b=b[order])
    np.savez(source/'test_scores.npz',source_ids=ids[order],generated_logits=np.zeros(n),log_ratio=np.zeros(n))
    np.savez(source/'candidates.npz',source_sample_index=np.ones(n,dtype=int),source_event_key=np.arange(n),
             truth_deltas=delta,deltas=delta[:,None])
    cfg=dict(classifier_directory=str(source),classifier_run='pinned',events=str(events),expected_events=n)
    # Dataset reader is an external boundary here; avoid importing local Lightning/torchvision.
    with patch.dict('sys.modules',{'scripts.sample_1110_cij':SimpleNamespace(validation_pool=lambda *a:pool)}),patch('scripts.diagnose_ztautau_cij.read_event_table',return_value=table):
        out,data=prepare(cfg)
    assert out['classifier_selected_epoch']==250
    np.testing.assert_array_equal(data['source_ids'],ids[order])
    np.testing.assert_allclose(data['condition'],cond[order])


def test_config_defaults_and_invalid_data():
    cfg=read_settings(ROOT/'config/conditional_tau_sampling_convergence.yaml')
    assert cfg['workers']==16 and cfg['batch_size']==1024 and cfg['classifier_run']=='pzq0nl1i'
    assert cfg['prefixes']==[1,2,4,8,16,32,64] and cfg['extend_run']=='ppoem0g9'
    t,g,s,w=fixture();s[0,0]=np.nan
    with pytest.raises(ValueError):convergence_report(t,g,s,w,bootstrap=20)


def test_full_endpoint_writes_reports_and_wandb_without_training(tmp_path):
    from scripts.run_tau_sampling_convergence import finish
    n=32;rng=np.random.default_rng(35);ids=np.array([str(i) for i in range(n)])
    va=np.tile([20.,5.,3.,18.],(n,1));vb=np.tile([18.,-7.,-3.,-15.],(n,1));kap=np.ones((n,2))
    delta=rng.normal(0,.03,(n,2,2)).astype('float32')
    tau,c,cij=physics_features(delta,va,vb,kap)
    a=dict(source_ids=ids,visible_a=va,visible_b=vb,kappas=kap,old_deltas=delta,old_logits=np.zeros(n),
        truth_tau=tau,truth_cij=cij,weight=np.ones(n),category=np.tile([11,22],16),visible_pt_sum=np.arange(n))
    np.savez(tmp_path/'inputs.npz',**a)
    for rank in range(16):
        pos=np.arange(rank,n,16)
        for j in range(8):
            np.savez(tmp_path/f'rank-{rank:02d}-candidate-{j:02d}.npz',positions=pos,source_ids=ids[pos],
                deltas=delta[pos],logits=np.zeros(len(pos)),tau=tau[pos],cij=cij[pos],replay_max_error=0.)
    cfg=dict(output=str(tmp_path),workers=16,prefixes=[1,2,4,8],bootstrap=20,seed=42,
             condition_pt_edges=[8,16,24],historical_cij_errors=[0.,0.],classifier_run='pinned',conditions=n)
    logs=[];saved=[]
    # Use the actual SDK summary, not dict: Summary.update accepts a mapping
    # only. A plain dict mock concealed the W&B 0.19.8 keyword-argument failure.
    from wandb.sdk.wandb_summary import Summary
    values={}
    summary=Summary(lambda:values)
    def update(record):
        for item in record.update:
            assert len(item.key)==1
            values[item.key[0]]=item.value
    summary._set_update_callback(update)
    run=SimpleNamespace(summary=summary,url='offline-test',log=lambda row:logs.append(row),save=lambda *a,**k:saved.append(a))
    fake=SimpleNamespace(Table=lambda **kw:kw,Image=lambda path:path)
    with patch.dict('sys.modules',{'wandb':fake}):finish(cfg,run)
    report=json.loads((tmp_path/'convergence_report.json').read_text())
    assert report['historical_K1']['error']==[0.,0.]
    assert run.summary['phase']=='complete' and run.summary['samples']==n*8
    assert run.summary['classifier_fits']==run.summary['policy_updates']==0
    assert (tmp_path/'COMPLETE').is_file() and len(saved)==2
    assert [row['candidates_per_condition'] for row in logs if 'candidates_per_condition' in row]==[1,2,4,8]
    assert any('Cij/conditional_mc_se_reweighted/kk' in row for row in logs)


@pytest.mark.parametrize('target',[32,64])
def test_extension_keeps_exact_nested_k8_reports_and_candidate_streams(tmp_path,target):
    from scripts.run_tau_sampling_convergence import new_candidate_indices, verify_inherited_report
    t,g,s,w=fixture()
    rng=np.random.default_rng(88)
    prefixes=[2**i for i in range(target.bit_length())]
    extended_g=np.concatenate((g,rng.normal(size=(len(w),target-8,9))),1)
    extended_s=np.concatenate((s,rng.normal(size=(len(w),target-8))),1)
    old=convergence_report(t,g,s,w,bootstrap=20)
    new=convergence_report(t,extended_g,extended_s,w,prefixes=prefixes,bootstrap=20)
    (tmp_path/'convergence_report.json').write_text(json.dumps(old))
    cfg=dict(extend_from=str(tmp_path),extend_run='old',inherited_candidates=8,prefixes=prefixes)
    assert list(new_candidate_indices(cfg))==list(range(8,target))
    verify_inherited_report(cfg,new)
    assert new['extension']['old_prefixes_reproduced'] is True
    assert [len(new['disjoint_panels'][str(k)]) for k in prefixes]==[target//k for k in prefixes]
    new['prefixes']['8']['reweighted'][0]+=1
    with pytest.raises(ValueError,match='Inherited K8'):verify_inherited_report(cfg,new)


@pytest.mark.parametrize('target',[32,64])
def test_extension_preflight_and_byte_preserving_shard_reuse(tmp_path,target):
    from scripts.run_tau_sampling_convergence import validate_extension,copy_inherited_shards
    source=tmp_path/'source';source.mkdir();output=tmp_path/'new';output.mkdir()
    n=35;ids=np.array([str(i) for i in range(n)])
    inputs=dict(source_ids=ids,weight=np.ones(n))
    np.savez(source/'inputs.npz',**inputs)
    old=dict(classifier_run='pzq0nl1i',classifier_checkpoint='/head.pt',generator_checkpoint='/raw.ckpt',
        events='/filtered/val',conditions=n,runtime='/runtime.yaml',workers=16,batch_size=1024,
        feature_batch_size=256,seed=930481,ddim_steps=20,packing_spec={'shapes':{}},
        weights='raw_state_dict_only',ratio_transform='raw',condition_pt_edges=[1,2,3],
        historical_cij_errors=[.5,1.4],bootstrap=20,prefixes=[1,2,4,8])
    (source/'manifest.json').write_text(json.dumps(old));(source/'COMPLETE').write_text('done')
    (source/'wandb.json').write_text('{"id":"ppoem0g9"}')
    (source/'convergence_report.json').write_text(json.dumps(dict(events=n,candidates=8)))
    for rank in range(16):
        pos=np.arange(rank,n,16)
        for j in range(8):
            np.savez(source/f'rank-{rank:02d}-candidate-{j:02d}.npz',positions=pos,source_ids=ids[pos],
                deltas=np.full((len(pos),2,2),j,dtype='float32'),logits=pos+j,
                tau=np.ones((len(pos),15)),cij=np.ones((len(pos),9)),replay_max_error=0.)
    cfg=dict(old,output=str(output),extend_from=str(source),extend_run='ppoem0g9',prefixes=[2**i for i in range(target.bit_length())])
    cfg=validate_extension(cfg,inputs)
    assert cfg['inherited_candidates']==8
    copy_inherited_shards(cfg)
    assert len(list(output.glob('rank-*.npz')))==128
    for f in source.glob('rank-*.npz'):assert f.read_bytes()==(output/f.name).read_bytes()
    with pytest.raises(ValueError,match='seed'):validate_extension(dict(cfg,seed=1),inputs)
    with pytest.raises(ValueError,match='inputs changed'):validate_extension(cfg,dict(inputs,weight=np.full(n,2.)))
    assert list(output.glob('*candidate-08.npz'))==[]
