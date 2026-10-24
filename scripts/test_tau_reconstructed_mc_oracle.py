"""Actual-MC ranking, fixed-energy export parity, refit and no-leakage checks."""
import copy
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'evenet_dgpo')]
from RL.DGPO_neutrino.tau_oracle_mixture import (RECO_MC_GEOMETRY,MC_TAU_COLUMNS,
    mix_reward,training_reward,directions,geometry_options,joint_contract,validate_resume_contract)
from scripts.train_tau_dgpo_general_oracle import settings,assemble

CFG=ROOT/'config/tau_dgpo_general_reconstructed_tau_oracle30_epoch287.yaml'


def opts():return geometry_options(settings(CFG)['joint_oracle'])


def actual_p4(truth,visible,magnitudes):
    p=directions(truth.double(),visible.double())*magnitudes[...,None]
    e=(magnitudes.square()+1.77686**2).sqrt()
    return torch.cat((e[...,None],p),-1)


def attach(batch,p4):
    for j,leg in enumerate('ab'):
        for i,q in enumerate(('E','px','py','pz')):
            batch[f'truth_tau_{leg}_{q}']=p4[:,j,i]


def test_actual_mc_momentum_changes_ranking_and_energy_term_is_constant():
    v=torch.tensor([[[1.,0.,0.],[1.,0.,0.]]],dtype=torch.float64)
    t=torch.zeros(1,2,2,dtype=torch.float64)
    d=torch.zeros(2,1,2,2,dtype=torch.float64)
    d[0,0,0,0]=.10;d[1,0,1,0]=.12
    p4=actual_p4(t,v,torch.tensor([[40.,5.]],dtype=torch.float64))
    s=torch.zeros(2,1,dtype=torch.float64)
    _,old=mix_reward(s,d,t,v,.3)
    r,new=mix_reward(s,d,t,v,.3,truth_tau_p4=p4,**opts())
    assert old['best'].item()==0 and new['best'].item()==1
    np.testing.assert_allclose(r.exp().numpy().ravel(),[.35,.65])
    changed=p4.clone();changed[...,0]+=10
    rr,other=mix_reward(s,d,t,v,.3,truth_tau_p4=changed,**opts())
    torch.testing.assert_close(r,rr)
    diff=other['distance']-new['distance']
    torch.testing.assert_close(diff[0],diff[1])


def test_cartesian_export_formula_mixture_and_permutation():
    rng=np.random.default_rng(42);n=25
    v=rng.normal(size=(n,2,3));t=rng.normal(size=(n,2,2))*.1
    d=rng.normal(size=(8,n,2,2))*.2;s=torch.tensor(rng.normal(size=(8,n)))
    mc=actual_p4(torch.tensor(t),torch.tensor(v),torch.tensor(rng.uniform(5,45,(n,2))))
    r,a=mix_reward(s,torch.tensor(d),torch.tensor(t),torch.tensor(v),.3,truth_tau_p4=mc,**opts())
    # Independent Cartesian form of exporter before post_calibrate_tau_tau.
    theta=np.arctan2(np.hypot(v[...,0],v[...,1]),v[...,2])+d[...,0]
    phi=np.arctan2(v[...,1],v[...,0])+d[...,1]
    p0=np.sqrt(45.6**2-1.777**2)
    reco=np.stack([np.full_like(theta,45.6),p0*np.sin(theta)*np.cos(phi),
                   p0*np.sin(theta)*np.sin(phi),p0*np.cos(theta)],-1)
    expected=((reco-mc.numpy())**2).sum(-1).mean(-1)/p0**2
    np.testing.assert_allclose(a['distance'],expected,rtol=2e-12,atol=1e-14)
    g=s.softmax(0);w=.7*g;w[a['best'],torch.arange(n)]+=.3
    torch.testing.assert_close(r.exp(),w)
    torch.testing.assert_close(w.sum(0),torch.ones(n,dtype=w.dtype))
    order=torch.tensor([7,2,4,0,1,6,3,5]);d[...,1]+=2*np.pi
    rr,_=mix_reward(s[order],torch.tensor(d)[order],torch.tensor(t),torch.tensor(v),.3,truth_tau_p4=mc,**opts())
    torch.testing.assert_close(rr,r[order])
    assert not r.requires_grad
    # No enforced back-to-back: the fixture's two reconstructed directions differ from -each other.
    assert np.max(np.linalg.norm(reco[...,0,1:]+reco[...,1,1:],axis=-1))>1


def test_missing_misaligned_unphysical_truth_and_changed_resume_fail():
    v=torch.ones(3,2,3);t=torch.zeros(3,2,2);d=torch.zeros(8,3,2,2);s=torch.zeros(8,3)
    mc=actual_p4(t,v,torch.ones(3,2)*40)
    for bad in (None,mc*float('nan'),mc.clone()):
        if bad is not None and torch.isfinite(bad).all():bad[...,1:]*=-1
        with pytest.raises(ValueError):mix_reward(s,d,t,v,.3,truth_tau_p4=bad,**opts())
    bad=mc.clone();bad[...,0]=1
    with pytest.raises(ValueError):mix_reward(s,d,t,v,.3,truth_tau_p4=bad,**opts())
    c=dict(enabled=True,alpha=.3,**settings(CFG)['joint_oracle'])
    p=dict(dgpo_checkpoint_version=1,dgpo_omnifold_reward_stack={'joint_oracle_contract':joint_contract(c)})
    validate_resume_contract(c,p);validate_resume_contract(c,dict(epoch=286,global_step=18655))
    for bad in (dict(c,alpha=.1),dict(c,fixed_energy_GeV=45.625),dict(c,reconstruction='post_calibrated')):
        with pytest.raises(ValueError):validate_resume_contract(bad,p)


def test_general_refit_restoration_and_truth_input_isolation():
    from scripts.test_tau_process_label_conditioning import bundle,batch
    from scripts.test_dgpo_tau_ratio import fake_hidden
    from RL.DGPO_neutrino.conditional_tau_reward import ConditionalTauReward
    from RL.DGPO_neutrino.dgpo_trainer import _dgpo_policy_conditioning_batch
    from scripts.train_conditional_spin_ratio import build_classifier
    torch.manual_seed(42);source=bundle();head=ConditionalTauReward(source,torch.nn.Linear(1,1),'cpu',2)
    b=batch();b['x_invisible']=torch.randn(5,2,2)*.03;b['x_invisible_mask']=torch.ones(5,2,1)
    v=torch.stack([torch.stack([b[f'lead_{leg}_visible_{q}'] for q in ('px','py','pz')],-1) for leg in 'ab'],1)
    attach(b,actual_p4(b['x_invisible'],v,torch.ones(5,2)*40))
    assert not set(MC_TAU_COLUMNS)&set(_dgpo_policy_conditioning_batch(b))
    d=torch.randn(8,5,2,2)*.04;d[2]=b['x_invisible']
    def check(model,draws,winner):
        s=model.compute(draws,b);r,m=training_reward(s,draws,b,.3,**opts())
        expected=.7*s.double().softmax(0);expected[winner,:]+=.3
        torch.testing.assert_close(r.double().exp(),expected,rtol=1e-6,atol=1e-7)
        assert m['oracle/alpha']==.3
        return s,r
    with patch('scripts.tau_backbone_alignment.candidate_hidden',side_effect=fake_hidden):
        before,_=check(head,d,2)
        poison=dict(b,**{key:torch.full_like(b[key],float('nan')) for key in MC_TAU_COLUMNS})
        torch.testing.assert_close(head.compute(d,poison),before)
        updated=copy.deepcopy(source['head']);updated.update(source_policy_step=500,optimizer_steps=2000)
        fitted=build_classifier(updated)
        with torch.no_grad():
            for p in fitted.parameters():p.add_(torch.randn_like(p)*.03)
        updated['state_dict']=fitted.state_dict();head.install(updated,policy_step=500,epoch=49)
        after,_=check(head,d,2);assert not torch.allclose(before.softmax(0),after.softmax(0))
        d[[2,6]]=d[[6,2]];_,expected=check(head,d,6)
        restored=ConditionalTauReward(source,torch.nn.Linear(1,1),'cpu',2)
        restored.load_stack_payload(head.stack_payload())
        _,actual=check(restored,d,6);torch.testing.assert_close(actual,expected)


def test_fresh_runtime_keeps_matched_training_setup():
    p=Path('/Users/yirenwu/Ztautau/ml_pipeline/artifacts/tau_step500_eval_config/source_runtime_step500.yaml')
    if not p.exists():pytest.skip('Local source runtime absent')
    source=yaml.safe_load(p.read_text());c=settings(CFG);r=assemble(source,c)
    control=assemble(source,settings(ROOT/'config/tau_dgpo_general_oracle30_epoch287.yaml'))
    assert r['network']==control['network']==source['network']
    for key in ('reference_trust','lr_schedule','K','advantage_estimator'):
        assert r['dgpo'][key]==control['dgpo'][key]
    assert r['dgpo']['tau_ratio']['fit']==control['dgpo']['tau_ratio']['fit']
    assert r['dgpo']['checkpoint_load_mode']=='weights_only'
    assert r['dgpo']['auto_resume_fallback_checkpoint_path'] is None
    assert r['platform']['batch_size']==512 and r['experiment']['policy_global_events']==8192
    assert [r['platform']['number_of_workers'],r['dgpo']['tau_ratio']['workers'],r['nersc']['execution']['workers']]==[16]*3
    assert r['nersc']['nodes']==4 and not r['nersc']['submit']
    assert r['dgpo']['tau_ratio']['refit_every_epochs']==50
    assert r['dgpo']['tau_ratio']['validation_every_epochs']==5
    assert len(r['dgpo']['tau_ratio']['joint_oracle_monitor']['edges'])==10
    assert r['logger']['wandb']['run_name']==c['run_name']
