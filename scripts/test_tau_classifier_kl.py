"""Classifier-KL label orientation, gradient identity and isolated updates."""
import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import yaml

from scripts.test_tau_full_trajectory import ChainPolicy, batch, replay_rng, generate_neutrino_candidates, DDIMSampler
from scripts.test_tau_trajectory_reward_refit import unbounded_bundle
from scripts.test_dgpo_tau_ratio import example_bundle
from scripts.test_tau_reward_transfer_launch import source_runtime, saved_state
from scripts import diagnose_tau_reward_mechanisms as launch
from scripts.tau_classifier_kl import validate_classifier_kl
from RL.DGPO_neutrino.tau_classifier_kl import paired_kl_panel, TauClassifierKL, ZeroLogRatio
from RL.DGPO_neutrino.tau_full_trajectory import FullTrajectoryController, slice_batch
from RL.DGPO_neutrino.conditional_tau_cycle import fit_head
from RL.DGPO_neutrino.conditional_tau_reward import ConditionalTauReward
from RL.DGPO_neutrino.tau_pathwise_reward import compute_reward_grad
from scripts.test_tau_full_trajectory import Trunk
from scripts.test_dgpo_tau_ratio import live_batch
from scripts.train_conditional_spin_ratio import build_classifier

OPTIONS = dict(anchor_step=1920, coefficient=1., refit_every_updates=1)
REFIT = dict(anchor_step=1920, ratio_bound=None, reference_recenter=True)


def config():
    return dict(method='pathwise', event_microbatch=2, parity_atol=1e-9, parity_rtol=1e-9,
                reference_coefficient=1., reward_refit=REFIT, classifier_kl=OPTIONS)


def test_protocol_disallows_stale_scaled_bounded_and_native_kl():
    assert validate_classifier_kl(OPTIONS) == OPTIONS
    for key, value in [('coefficient', 2), ('coefficient', True), ('refit_every_updates', 5), ('anchor_step', 1880)]:
        with pytest.raises(ValueError):
            validate_classifier_kl({**OPTIONS, key:value})
    for values in ({'method':'native'}, {'reward_refit':None},
                   {'reward_refit':{**REFIT,'ratio_bound':30}}, {'reference_balance':{}}):
        with pytest.raises(ValueError):
            FullTrajectoryController({**config(), **values})
    controller = FullTrajectoryController(config())
    assert controller.reference_coefficient == 1 and controller.native_reference_coefficient == 0


def test_pairing_current_positive_anchor_negative_without_truth_replacement():
    truth = np.full((3,4), 99.)
    panel = dict(candidate_truth=truth, condition=np.zeros((3,2)), event_weight=np.ones(3), split=np.arange(3))
    current, anchor = np.ones_like(truth), -np.ones_like(truth)
    pos, neg = paired_kl_panel(panel, current, anchor)
    assert pos['candidate_truth'] is current and neg['features'] is anchor
    assert panel['candidate_truth'] is truth
    assert pos['condition'] is panel['condition'] and pos['split'] is panel['split']
    with pytest.raises(ValueError):
        paired_kl_panel(panel, current[:2], anchor)


def test_zero_ratio_has_exactly_zero_input_derivative():
    x=torch.randn(3,7,requires_grad=True)
    value=ZeroLogRatio()(torch.zeros(3,2),x)
    gradient,=torch.autograd.grad(value.sum(),x)
    assert value.count_nonzero()==0 and gradient.count_nonzero()==0


def test_actual_frozen_kl_head_forward_and_input_gradient_match_value_oracle():
    reward=ConditionalTauReward(unbounded_bundle(),Trunk(),'cpu')
    kl=copy.copy(reward)
    saved=copy.deepcopy(reward.installed_head)
    torch.manual_seed(23)
    kl.head=build_classifier(saved).eval().requires_grad_(False)
    kl.installed_head=saved
    x=(torch.randn(2,3,2,2)*.2).requires_grad_()
    b=live_batch(3)
    oracle=kl.compute(x.detach(),b)
    score=compute_reward_grad(kl,x,b)
    torch.testing.assert_close(score,oracle,rtol=3e-5,atol=2e-6)
    gradient,=torch.autograd.grad(score.mean(),x)
    assert torch.isfinite(gradient).all() and gradient.norm()>0
    assert all(p.grad is None for p in kl.head.parameters())
    assert all(p.grad is None for p in kl.backbone.parameters())
    assert reward.head is not kl.head


def test_gaussian_ratio_objective_and_full_ddim_vjp_align_without_velocity():
    model, b = ChainPolicy(), batch(3)
    sampler=DDIMSampler(torch.device('cpu'),x0_mode='stable_v',dtype=torch.float64)
    controller=FullTrajectoryController(config())
    kl_source=object()
    controller.classifier_kl=SimpleNamespace(source=kl_source,fit_step=1921)
    # Exact unit-variance Gaussian logits: truth/anchor and current/anchor.
    # Their difference's endpoint gradient is current_mean - truth_mean;
    # the arbitrary anchor cancels identically.
    def score(source,x,batch,**kwargs):
        mean=.3 if source is kl_source else .2
        return (-.5*(x-mean)**2+.5*(x+1.7)**2).sum((-2,-1))
    torch.manual_seed(51)
    candidates=controller.rollout(model,b,sampler,K=8,num_ddim_steps=20,device=torch.device('cpu'))
    rewards=score(None,candidates,b)
    loss=0
    for k,start,stop,state in controller.records:
        sub=slice_batch(b,start,stop)
        with replay_rng(state,torch.device('cpu')):
            generated=generate_neutrino_candidates(model,sub,sampler,K=1,num_ddim_steps=20,
                device=torch.device('cpu'),differentiable=True)
            loss=loss+(score(kl_source,generated,sub)-score(None,generated,sub)).sum()/(8*3)
    expected,=torch.autograd.grad(loss,model.gains)
    with patch('RL.DGPO_neutrino.tau_full_trajectory.compute_reward_grad',side_effect=score):
        metrics=controller.backward(model,b,sampler,None,candidates,rewards,device=torch.device('cpu'))
    torch.testing.assert_close(model.gains.grad,expected,rtol=1e-9,atol=1e-9)
    assert metrics['trajectory/classifier_kl/velocity_penalty_enabled']==0
    assert 'trajectory/local_reward_to_reference_gradient_ratio' not in metrics
    assert np.isclose(metrics['trajectory/objective_loss'],loss.detach().item())


def test_refresh_labels_and_preserves_reward_reference_actor_and_optimizer(tmp_path):
    reward=SimpleNamespace(installed_head=unbounded_bundle()['head'],head=torch.nn.Linear(3,1),round_id=21)
    actor=torch.nn.Linear(2,2)
    ref=copy.deepcopy(actor)
    optimizer=torch.optim.AdamW(actor.parameters())
    actor(torch.ones(2,2)).sum().backward();optimizer.step()
    before=copy.deepcopy(actor.state_dict());opt=copy.deepcopy(optimizer.state_dict())
    truth=np.full((4,3),99.,dtype=np.float32)
    panel=dict(candidate_truth=truth,source_ids=np.arange(4),condition=np.zeros((4,2)),split=np.arange(4)%2,event_weight=np.ones(4))
    current=np.ones((4,3),dtype=np.float32);anchor=-current
    cycle=SimpleNamespace(train=panel,actor=actor,reference=ref,reward=reward,
        cfg=dict(refit_seed=42),device=torch.device('cpu'),rank=0,world=1,emit=lambda *args:None)
    cycle.generate=lambda *args:dict(features=current)
    calls=[]
    def fit(pos,neg,folder,**kwargs):
        assert pos['candidate_truth'] is current and neg['features'] is anchor
        assert kwargs['min_selected_steps']==1000 and kwargs['reward_ratio_bound'] is None
        calls.append(kwargs)
        return torch.nn.Linear(3,1),dict(reward.installed_head,source_policy_step=1921,min_steps=1000),dict(minimum_fit_steps_met=True,total_steps=1100,best_steps=1050),None
    cycle.fit=fit
    critic=TauClassifierKL(cycle,OPTIONS,tmp_path,anchor_directory=tmp_path/'anchor')
    critic.anchor=anchor
    assert isinstance(critic.refresh(1920).head,ZeroLogRatio)
    critic.refresh(1921)
    assert len(calls)==1 and cycle.reward is reward and reward.round_id==21
    assert critic.source is not reward and critic.source.head is not reward.head
    for k,v in before.items():torch.testing.assert_close(actor.state_dict()[k],v,rtol=0,atol=0)
    assert optimizer.state_dict()['param_groups']==opt['param_groups']
    for p,s in opt['state'].items():
        for k,v in s.items():torch.testing.assert_close(optimizer.state_dict()['state'][p][k],v,rtol=0,atol=0)
    with pytest.raises(ValueError,match='sequential'):
        critic.refresh(1921)
    assert critic.checkpoint_payload()['report']['negative_class']=='fixed_step1920_anchor'
    cycle.fit=lambda *a,**k:(torch.nn.Linear(3,1),dict(reward.installed_head,source_policy_step=1922,min_steps=1000),
        dict(minimum_fit_steps_met=True,total_steps=1100,best_steps=700),None)
    with pytest.raises(ValueError,match='fit contract'):
        critic.refresh(1922)
    assert critic.fit_step==1921


def test_fit_selection_floor_applies_to_selected_weights(tmp_path):
    cfg=example_bundle()['head']
    cfg.update(condition_dim=2,candidate_dim=21,hidden=8,head_depth=1,cross_attention=False,
        lr=.001,epochs=3,batch_size=8,min_lr=.0001,patience=25,min_delta=.0001,
        weight_decay=0.,seed=42,min_steps=4,min_selected_steps=4,ratio_bound=None)
    data=dict(condition=np.zeros((32,2),dtype=np.float32),candidate_truth=np.ones((32,21),dtype=np.float32),
        candidate_generated=-np.ones((32,21),dtype=np.float32),event_weight=np.ones(32),split=np.r_[np.zeros(24),np.ones(8)])
    metrics=iter([dict(bce=.1,auc=.6),dict(bce=.3,auc=.6),dict(bce=.4,auc=.6)])
    with patch('scripts.train_conditional_spin_ratio.pair_metrics',side_effect=lambda *a:next(metrics)):
        _,saved,status=fit_head(cfg,data,tmp_path/'best.pt',torch.device('cpu'),0,1,lambda *a:None)
    assert status['best_steps']==6 and saved['optimizer_steps']==6


def test_real_config_disables_velocity_and_preserves_native_optimizer_contract(tmp_path):
    settings=yaml.safe_load((Path(__file__).resolve().parents[1]/'config/tau_full_trajectory_1920_mb64_classifier_kl.yaml').read_text())
    original=source_runtime();before=copy.deepcopy(original)
    configured=launch.configure(original,settings,launch.source_metadata(saved_state()),pinned=tmp_path/'source.ckpt',
        output=tmp_path,stage='trajectory',method='pathwise')
    assert original==before
    assert configured['dgpo']['reference_trust']['enabled'] is False
    assert configured['dgpo']['reference_trust']['coefficient']==0
    assert configured['experiment']['classifier_kl']==OPTIONS
    assert 'classifier KL' in configured['logger']['wandb']['run_name']
    with pytest.raises(ValueError):
        launch.configure(original,settings,launch.source_metadata(saved_state()),pinned=tmp_path/'source.ckpt',output=tmp_path,stage='trajectory',method='native')
