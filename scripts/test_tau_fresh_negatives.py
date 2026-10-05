"""CPU protocol/actual worker tests; CUDA/DDIM production smoke test is user-launched."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from scripts.tau_fresh_negatives import isolated_rng, draw_seed, candidate_features
from scripts.run_tau_fresh_negatives import read_settings, make_config, verify_matched_film, panel_report, ROOT
from scripts.train_conditional_spin_ratio import paired_loss, training_worker
from scripts.test_tau_conditioning import model_config


def test_rng_precision_isolation_and_new_streams():
    torch.manual_seed(42)
    expected = torch.rand(5)
    torch.manual_seed(42)
    precision = torch.get_float32_matmul_precision()
    with isolated_rng('cpu',100):
        first = torch.rand(5)
        torch.set_float32_matmul_precision('medium')
    assert torch.get_float32_matmul_precision() == precision
    torch.testing.assert_close(torch.rand(5),expected,atol=0,rtol=0)
    with isolated_rng('cpu',100):
        torch.testing.assert_close(torch.rand(5),first,atol=0,rtol=0)
    seeds = [draw_seed(1930930,e,r,b) for e in range(250) for r in range(16) for b in range(22)]
    assert len(seeds)==len(set(seeds))
    assert draw_seed(1,0,0,0)!=draw_seed(1,1,0,0)
    with pytest.raises(ValueError): draw_seed(1,0,16,0)
    with pytest.raises(RuntimeError):
        with isolated_rng('cpu',1):
            torch.set_float32_matmul_precision('medium')
            raise RuntimeError('fixture')
    assert torch.get_float32_matmul_precision()==precision


def test_features_share_frozen_preprocessing_without_truth_or_spin_input():
    from scripts.diagnose_ztautau_cij import tau_from_deltas
    from scripts.tau_relative_inputs import relative_angles
    from scripts.train_conditional_spin_ratio import tau_features
    va = np.tile([10.,1.,2.,3.],(5,1));vb = np.tile([12.,-2.,1.,4.],(5,1))
    d = torch.zeros(5,2,2)
    stats = dict(mean=[.1]*6,scale=[2.]*6)
    calls=[]
    def hidden(policy,batch,delta):
        assert not torch.is_grad_enabled()
        calls.append(len(delta))
        return np.ones((len(delta),8),np.float32)
    with isolated_rng('cpu',0),patch('scripts.tau_backbone_alignment.candidate_hidden',side_effect=hidden):
        output = candidate_features(None,{'x':torch.zeros(5,1)},d,va,vb,stats,2)
    a,b=tau_from_deltas(va,d[:,0].numpy()),tau_from_deltas(vb,d[:,1].numpy())
    expected=np.concatenate((np.ones((5,8)),(relative_angles(a,b,va,vb)-.1)/2,tau_features(a,b)),1)
    np.testing.assert_allclose(output,expected,atol=1e-7)
    assert calls==[2,2,1] and output.shape==(5,29)
    assert stats==dict(mean=[.1]*6,scale=[2.]*6)


def test_balanced_bce_needs_no_ratio_prior_correction():
    class Constant(torch.nn.Module):
        def forward(self,c,t): return t[:,0]
    p,q=torch.full((4,1),.4),torch.full((4,1),-.7)
    weights=torch.tensor([.2,.8,1.,2.])
    loss,_,_=paired_loss(Constant(),torch.zeros(4,1),p,q,weights)
    expected=(torch.nn.functional.softplus(-p[:,0])+torch.nn.functional.softplus(q[:,0]))/2
    torch.testing.assert_close(loss,(expected*weights).mean())


def test_config_preserves_historical_optimizer_and_architecture(tmp_path):
    settings=read_settings(ROOT/'config/conditional_tau_fresh_negatives_10pct.yaml')
    settings.update(fresh_runtime='runtime',fresh_generator_checkpoint='raw1110')
    base=dict(seed=42,lr=2e-4,min_lr=1e-5,epochs=250,hidden=128,dropout=.05,
        weight_decay=.001,workers=16,batch_size=1024,head_kind='film',head_depth=3,
        patience=25,min_delta=1e-4,min_steps=1000,relative_dim=6,ratio_objective='bce',mmd_coefficient=0.)
    with patch('scripts.run_tau_fresh_negatives.prepare_fit_inputs',return_value='fit.npz'):
        cfg=make_config(settings,base,Path('/baseline'),tmp_path,'fresh')
    assert all(cfg[k]==v for k,v in base.items())
    assert cfg['fresh_negatives'] and cfg['policy_updates']==0 and cfg['backbone_updates']==0
    verify_matched_film(base,base)
    with pytest.raises(ValueError,match='lr'):
        verify_matched_film(dict(base,lr=1e-3),base)


def test_panel_identical_heads_produce_zero_difference():
    rng=np.random.default_rng(12);n,k=24,4
    inputs=dict(weight=rng.uniform(.5,2,n),truth_cij=rng.normal(size=(n,9)))
    data=dict(logits=rng.normal(size=(n,k)),cij=rng.normal(size=(n,k,9)))
    report=panel_report(inputs,data,data['logits'].copy(),20,42)
    for label in ('fresh_raw_minus_old_raw','fresh_cap30_minus_old_cap30'):
        assert report['comparisons'][label]['error_change']==pytest.approx(0,abs=1e-14)
        np.testing.assert_allclose(report['comparisons'][label]['error_change_ci95'],0,atol=1e-14)
    with pytest.raises(ValueError): panel_report(inputs,data,np.zeros((n,k+1)),20,42)


def test_actual_training_refreshes_fit_only_and_keeps_val_fixed(tmp_path):
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
        skip_train_mmd_diagnostic=True,fresh_negatives=True)
    draws=[];validations=[];reports=[]
    class FakeRefresh:
        def __init__(self,cfg,device,rank,old):
            np.testing.assert_array_equal(old,data['candidate_generated'][:32])
            self.parity=0.
        def begin_epoch(self,epoch): self.events=0;self.seconds=0.
        def draw(self,idx,epoch,batch):
            draws.append((epoch,idx.copy()))
            self.events+=len(idx)
            return torch.as_tensor(data['candidate_generated'][idx]+epoch+1)
    from scripts.train_conditional_spin_ratio import score_pair
    def score(model,c,t,g,device):
        validations.append(g.copy())
        return score_pair(model,c,t,g,device)
    with patch('scripts.tau_fresh_negatives.FreshNegatives',FakeRefresh), \
         patch('scripts.train_conditional_spin_ratio.score_pair',side_effect=score), \
         patch('ray.train.get_context',return_value=SimpleNamespace(get_world_rank=lambda:0,get_world_size=lambda:1)), \
         patch('ray.train.torch.get_device',return_value=torch.device('cpu')), \
         patch('ray.train.torch.prepare_model',side_effect=lambda m:m), \
         patch('ray.train.report',side_effect=reports.append), \
         patch('torch.distributed.all_reduce'),patch('torch.distributed.broadcast'):
        training_worker(cfg)
    assert len(draws)==8 and len(reports)==2
    for epoch in range(2):
        np.testing.assert_array_equal(np.sort(np.concatenate([idx for e,idx in draws if e==epoch])),np.arange(32))
        np.testing.assert_array_equal(validations[epoch],data['candidate_generated'][32:])
        assert reports[epoch]['fresh_negative_events']==32
        assert reports[epoch]['fresh_validation_refreshed']==0
    assert reports[-1]['optimizer_steps']==8


def test_prepare_aligns_only_fit_rows_and_rejects_changed_condition(tmp_path):
    from scripts.run_tau_fresh_negatives import prepare_fit_inputs
    raw=np.array([[1.],[2.],[3.]],np.float32)
    truth=np.zeros((3,2,2),np.float32)
    pool=dict(condition=torch.from_numpy(raw),truth=torch.from_numpy(truth),
        source_sample_index=torch.zeros(3,dtype=torch.int64),source_event_key=torch.arange(10,13),
        rows_read=416701,invalid_target_rows=0)
    onehot=np.eye(16,dtype=np.float32)[[0,0,0]]
    condition=np.concatenate((raw[::-1],onehot),1)
    prepared=dict(condition=condition,split=np.array([0,1,0]),source_ids=np.array(['0:12','0:11','0:10']),
        category=np.array([11,11,11]),condition_mean=np.zeros(17),condition_scale=np.ones(17),
        event_weight=np.ones(3))
    np.savez(tmp_path/'prepared.npz',**prepared)
    np.savez(tmp_path/'candidates.npz',source_sample_index=np.zeros(3,dtype=np.int64),
        source_event_key=np.arange(10,13),truth_deltas=truth,deltas=np.arange(12,dtype=np.float32).reshape(3,1,2,2))
    columns=dict(source_sample_index=np.zeros(3,dtype=np.int64),source_event_key=np.arange(10,13),
        event_category=np.full(3,11),event_weight=np.ones(3))
    for leg in ('a','b'):
        for j,c in enumerate(('E','px','py','pz')): columns[f'lead_{leg}_visible_{c}']=np.full(3,j+1.)
    table={k:SimpleNamespace(to_numpy=lambda v=v:v) for k,v in columns.items()}
    cfg=dict(prepared=str(tmp_path/'prepared.npz'),train_events='filtered',packing_spec={},train_source=str(tmp_path))
    # This test verifies alignment, not the unrelated Lightning/CUDA parquet reader.
    with patch.dict('sys.modules',{'scripts.sample_1110_cij':SimpleNamespace(validation_pool=lambda *args:pool)}), \
         patch('scripts.diagnose_ztautau_cij.read_event_table',return_value=table), \
         patch('scripts.conditional_tau_preprocessing.apply_masked_feature',side_effect=lambda raw,*args:raw):
        path=prepare_fit_inputs(cfg,tmp_path)
        with np.load(path) as f:
            np.testing.assert_array_equal(f['source_ids'],['0:12','0:10'])
            np.testing.assert_array_equal(f['raw_condition'],[[3.],[1.]])
            assert f['old_deltas'][0,0,0]==8
        prepared['condition']=condition+1
        np.savez(tmp_path/'prepared.npz',**prepared)
        with pytest.raises(ValueError,match='preprocessing'):
            prepare_fit_inputs(cfg,tmp_path)
