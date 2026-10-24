import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'evenet_dgpo')]
from RL.DGPO_neutrino.tau_adaptive_oracle import DEFAULT_GATE,paired_auc,next_state,live_alpha


def test_paired_auc_matches_exact_ties_and_includes_pair_covariance():
    rng=np.random.default_rng(42);p=rng.integers(0,8,100);q=rng.integers(0,8,100)
    a=paired_auc(p,q)
    assert a['auc']==pytest.approx(((p[:,None]>q).astype(float)+.5*(p[:,None]==q)).mean())
    exact=paired_auc(p,p)
    assert exact['auc']==.5 and exact['auc_se']<1e-16
    assert paired_auc(p,q[::-1])['auc']==a['auc']
    assert paired_auc(p,q[::-1])['auc_se']!=pytest.approx(a['auc_se'])


def test_gate_drops_to_zero_and_does_not_reset_baseline_on_refit():
    s=next_state(DEFAULT_GATE,None,dict(auc=.8,auc_se=.01,events=1000),round_id=0,step=0)
    assert s['alpha']==.3
    s1=next_state(DEFAULT_GATE,s,dict(auc=.65,auc_se=.01,events=1000),round_id=1,step=500)
    assert 0<s1['alpha']<.3 and s1['baseline_gap']==s['baseline_gap']
    s2=next_state(DEFAULT_GATE,s1,dict(auc=.51,auc_se=.01,events=1000),round_id=2,step=1000)
    assert s2['alpha']==0
    s3=next_state(DEFAULT_GATE,s2,dict(auc=.9,auc_se=.01,events=1000),round_id=3,step=1500)
    assert s3['alpha']==.3  # recovers with mismatch; no hardcoded monotonic decay
    with pytest.raises(ValueError,match='inverted'):
        next_state(DEFAULT_GATE,s,dict(auc=.3,auc_se=.01),round_id=1,step=500)
    with pytest.raises(ValueError,match='round mismatch'):
        next_state(DEFAULT_GATE,s,dict(auc=.7,auc_se=.01),round_id=3,step=500)


def test_live_state_resume_round_and_cap_fail_closed():
    cfg=dict(alpha=.3,adaptive=DEFAULT_GATE)
    state=next_state(DEFAULT_GATE,None,dict(auc=.8,auc_se=.01),round_id=0,step=0)
    reward=SimpleNamespace(round_id=0,denominator_step=0,adaptive_oracle_state=json.loads(json.dumps(state)))
    assert live_alpha(cfg,reward)==.3
    reward.round_id=1
    with pytest.raises(ValueError,match='stale'):live_alpha(cfg,reward)
    with pytest.raises(ValueError,match='as fixed'):live_alpha(dict(alpha=.3),reward)
    assert live_alpha(dict(alpha=.1),SimpleNamespace())==.1


def test_matched_config_and_legacy_runtime_unchanged():
    from scripts.train_tau_dgpo_general_oracle import settings,assemble
    fixed=settings(ROOT/'config/tau_dgpo_general_oracle30_epoch287.yaml')
    adapt=settings(ROOT/'config/tau_dgpo_general_adaptive_oracle30_epoch287.yaml')
    source=Path('/Users/yirenwu/Ztautau/ml_pipeline/artifacts/tau_step500_eval_config/source_runtime_step500.yaml')
    if not source.exists():pytest.skip('Source unavailable')
    source=yaml.safe_load(source.read_text());a,b=assemble(source,fixed),assemble(source,adapt)
    assert b['dgpo']['oracle_mixture'].pop('adaptive')==DEFAULT_GATE
    assert b['dgpo']['tau_ratio'].pop('adaptive_oracle')==DEFAULT_GATE
    # Runtime paths differ between arms, while actual optimizer/sampling/fit settings match.
    def replace_paths(x):
        if isinstance(x,dict):return {k:replace_paths(v) for k,v in x.items()}
        if isinstance(x,list):return [replace_paths(v) for v in x]
        if isinstance(x,str):return x.replace(adapt['output'],fixed['output'])
        return x
    assert a['dgpo']==replace_paths(b['dgpo'])
    assert a['network']==b['network'] and a['platform']==b['platform']
    assert 'adaptive_oracle30' in b['nersc']['execution']['command']
    assert a['dgpo']['auto_resume_from_last']


def test_initial_calibration_once_resume_and_refit_commit(tmp_path,monkeypatch):
    from RL.DGPO_neutrino.conditional_tau_cycle import ConditionalTauCycle
    from RL.DGPO_neutrino.conditional_tau_reward import POLICY_CONDITIONING_CONTRACT
    c=object.__new__(ConditionalTauCycle)
    c.cfg=dict(adaptive_oracle=DEFAULT_GATE,refit_seed=42,wandb_profile=None)
    c.rank=0;c.world=1;c.output=tmp_path;c.device=torch.device('cpu');c.emit=lambda *a:None
    c.actor=torch.nn.Linear(1,1);c.reference=torch.nn.Linear(1,1)
    reward=SimpleNamespace(round_id=0,denominator_step=0,last_refit_epoch=-1,adaptive_oracle_state=None,head=object(),policy_conditioning_contract=POLICY_CONDITIONING_CONTRACT)
    c.reward=reward;calls=[]
    def calibrate(model,step,round_id):
        calls.append((step,round_id))
        return next_state(DEFAULT_GATE,reward.adaptive_oracle_state,dict(auc=.8 if step==0 else .6,auc_se=.01),round_id=round_id,step=step)
    c.calibrate_adaptive_oracle=calibrate
    c.initialize_adaptive_oracle(0);c.initialize_adaptive_oracle(250)
    assert calls==[(0,0)]
    c.train=dict(source_ids=np.arange(416701));c.validation=dict(source_ids=np.arange(500000,500020))
    c.generate=lambda *a,**kw:dict(features=np.zeros((1,1)))
    c.fit=lambda *a,**kw:(object(),dict(optimizer_steps=1000),dict(minimum_fit_steps_met=True),None)
    def install(best,policy_step,epoch):
        reward.round_id+=1;reward.denominator_step=policy_step;reward.last_refit_epoch=epoch
    reward.install=install
    c.refit(49,500)
    assert calls==[(0,0),(500,1)] and reward.adaptive_oracle_state['alpha']<.3
    assert live_alpha(dict(alpha=.3,adaptive=DEFAULT_GATE),reward)==reward.adaptive_oracle_state['alpha']
    c.initialize_adaptive_oracle(500)
    assert len(calls)==2
    reward.adaptive_oracle_state=None
    with pytest.raises(ValueError,match='missing'):c.initialize_adaptive_oracle(500)


def test_adaptive_state_roundtrips_in_real_reward_stack():
    from scripts.test_dgpo_tau_ratio import example_bundle
    from RL.DGPO_neutrino.conditional_tau_reward import ConditionalTauReward
    bundle=example_bundle();r=ConditionalTauReward(bundle,torch.nn.Linear(1,1),'cpu')
    r.adaptive_oracle_state=next_state(DEFAULT_GATE,None,dict(auc=.8,auc_se=.01),round_id=0,step=0)
    clone=ConditionalTauReward(bundle,torch.nn.Linear(1,1),'cpu')
    clone.load_stack_payload(r.stack_payload())
    assert clone.adaptive_oracle_state==r.adaptive_oracle_state
    assert live_alpha(dict(alpha=.3,adaptive=DEFAULT_GATE),clone)==.3
    # Legacy stacks still load, but adaptive use of a legacy stack is forbidden.
    legacy=r.stack_payload();legacy.pop('adaptive_oracle_state');clone.load_stack_payload(legacy)
    with pytest.raises(ValueError,match='Missing'):live_alpha(dict(alpha=.3,adaptive=DEFAULT_GATE),clone)


def test_calibration_reuses_fixed_event_ids_and_logs_raw_head_scores(tmp_path,monkeypatch):
    from RL.DGPO_neutrino.conditional_tau_cycle import ConditionalTauCycle
    c=object.__new__(ConditionalTauCycle);n=12010;rng=np.random.default_rng(4)
    c.cfg=dict(adaptive_oracle=DEFAULT_GATE,fit=dict(batch_size=1024));c.output=tmp_path;c.rank=0;c.world=1;c.device=torch.device('cpu')
    c.reward=SimpleNamespace(adaptive_oracle_state=None);logs=[];c.emit=lambda x,step:logs.append(x)
    truth=rng.normal(1,1,(n,1)).astype('float32')
    c.validation=dict(source_ids=np.arange(n),condition=np.zeros((n,1),dtype='float32'),candidate_truth=truth)
    seen=[]
    def generate(panel,out,k,seed,physics):
        out.mkdir(parents=True,exist_ok=True);seen.append(panel['source_ids'].copy())
        assert k==1 and seed==DEFAULT_GATE['generation_seed']
        return dict(features=np.random.default_rng(5).normal(0,1,(len(panel['source_ids']),1)).astype('float32'))
    c.generate=generate
    class Head(torch.nn.Module):
        def forward(self,c,x):return x[:,0]
    s=c.calibrate_adaptive_oracle(Head(),0,0);c.reward.adaptive_oracle_state=s
    s2=c.calibrate_adaptive_oracle(Head(),500,1)
    np.testing.assert_array_equal(seen[0],seen[1]);assert s2['baseline_gap']==s['baseline_gap']
    assert len(seen[0])==11903 and len(logs)==2
    assert (tmp_path/'oracle-calibration-step-00000500/gate.json').exists()
