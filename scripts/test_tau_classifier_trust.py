"""Real AdamW transactions, held-out estimator and accepted-update clocks."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from scripts.tau_classifier_kl import validate_classifier_trust
from scripts.test_tau_classifier_kl import OPTIONS, REFIT, config, unbounded_bundle
from scripts.test_tau_reward_transfer_launch import source_runtime, saved_state
from scripts import diagnose_tau_reward_mechanisms as launch
from RL.DGPO_neutrino.tau_classifier_kl import TauClassifierKL
from RL.DGPO_neutrino.tau_classifier_trust import TauClassifierTrust, kl_confidence_bound
from RL.DGPO_neutrino.tau_full_trajectory import FullTrajectoryController
from RL.DGPO_neutrino.tau_reward_transfer import TauRewardTransferProbe

TRUST = dict(anchor_step=1920, max_kl=.01, backtrack_factor=.5, max_backtracks=4,
             confidence=.95, validation_seed=20261005, evaluation_candidates=1)
ROOT = Path(__file__).resolve().parents[1]


def assert_state_equal(left, right):
    assert left['param_groups'] == right['param_groups']
    assert left['state'].keys() == right['state'].keys()
    for p, state in left['state'].items():
        for k, value in state.items():
            torch.testing.assert_close(value, right['state'][p][k], rtol=0, atol=0)


def fixture(tmp_path):
    torch.manual_seed(83)
    actor = torch.nn.Linear(2,1).double()
    actor.register_buffer('normalizer', torch.tensor([2.]))
    optimizer = torch.optim.AdamW(actor.parameters(), lr=.1, weight_decay=.03)
    actor(torch.ones(3,2,dtype=torch.float64)).square().mean().backward()
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i:.98**i)
    actor(torch.ones(3,2,dtype=torch.float64)).square().mean().backward()
    panel = dict(source_ids=np.arange(4), candidate_truth=np.zeros((4,3),dtype=np.float32))
    cycle = SimpleNamespace(actor=actor, rank=0, world=1, device=torch.device('cpu'),
        cfg=dict(refit_seed=42), train=panel, emit=lambda *args:None,
        reward=SimpleNamespace(installed_head=unbounded_bundle()['head']))
    critic = TauClassifierKL(cycle, OPTIONS, tmp_path/'kl', anchor_directory=tmp_path/'anchor')
    critic.anchor = np.zeros((4,3),dtype=np.float32)
    critic.refresh(1920)
    fits = []
    def fit(step, folder, **kwargs):
        fits.append((step, folder, kwargs, copy.deepcopy(actor.state_dict())))
        return (torch.nn.Linear(3,1).eval().requires_grad_(False),
            dict(cycle.reward.installed_head, source_policy_step=step, min_steps=1000),
            dict(minimum_fit_steps_met=True,total_steps=1100,best_steps=1050))
    critic.fit_candidate = fit
    guard = TauClassifierTrust(critic, TRUST, tmp_path/'trust')
    return actor, optimizer, scheduler, critic, guard, fits


def test_protocol_requires_classifier_kl_and_rejects_ambiguous_boundaries():
    assert validate_classifier_trust(TRUST) == TRUST
    for key, value in [('max_kl',0), ('max_kl',float('nan')), ('max_kl',True),
                       ('anchor_step',1880), ('max_backtracks',True), ('max_backtracks',11),
                       ('backtrack_factor',1), ('confidence',1), ('evaluation_candidates',8)]:
        with pytest.raises(ValueError):
            validate_classifier_trust({**TRUST,key:value})
    with pytest.raises(ValueError,match='coefficient-one'):
        FullTrajectoryController({**config(), 'classifier_kl':None, 'hard_trust_region':TRUST})


def test_bound_is_mean_log_ratio_with_event_uncertainty_not_per_event_score():
    scores=np.array([-.02,.01,.025,.004])
    weights=np.array([1.,2.,1.,3.])
    stats=kl_confidence_bound(scores, weights, confidence=.95, trials=5)
    assert stats['mean'] == pytest.approx(np.average(scores,weights=weights))
    assert stats['upper'] > stats['mean'] and stats['standard_error'] > 0
    assert stats['critical_value'] > 1.645
    plain=kl_confidence_bound(scores,weights,confidence=.95,trials=1)
    assert stats['upper'] > plain['upper']
    negative=kl_confidence_bound(np.full(5,-.1),np.ones(5),confidence=.95,trials=5)
    assert negative['upper'] < 0 and not negative['valid_nonnegative_kl']
    for values, w in [(scores[:1],weights[:1]), (scores,np.array([0.,0.,0.,0.])),
                       (np.array([float('nan')]*4),weights)]:
        with pytest.raises(ValueError):
            kl_confidence_bound(values,w,confidence=.95,trials=5)


def test_backtracking_matches_same_adamw_step_at_scaled_lr_and_reuses_exact_head(tmp_path):
    actor,opt,scheduler,critic,guard,fits=fixture(tmp_path)
    expected=copy.deepcopy(actor)
    expected_opt=torch.optim.AdamW(expected.parameters(),lr=.1,weight_decay=.03)
    expected_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
    for p,q in zip(actor.parameters(),expected.parameters()):q.grad=p.grad.clone()
    expected_opt.param_groups[0]['lr'] *= .25
    expected_opt.step()
    expected_opt.param_groups[0]['lr'] = opt.param_groups[0]['lr']
    answers=iter([.08,.02,.005])
    def evaluate(fitted,folder,step):
        value=next(answers)
        return dict(mean=value,standard_error=0,upper=value,accepted=value<=.01)
    guard._evaluate=evaluate
    schedule_before=copy.deepcopy(scheduler.state_dict())
    accepted,metrics=guard.propose(actor,opt,1920)
    assert accepted and metrics['tau/classifier_trust/accepted_scale']==.25 and len(fits)==3
    for k,v in expected.state_dict().items():
        torch.testing.assert_close(actor.state_dict()[k],v,rtol=0,atol=1e-15)
    assert_state_equal(opt.state_dict(),expected_opt.state_dict())
    assert scheduler.state_dict()==schedule_before  # caller owns the one accepted tick
    cached=critic.pending_fit[1]
    critic.fit_candidate=lambda *a,**k:pytest.fail('Accepted-policy head must be reused')
    critic.refresh(1921)
    assert critic.source.head is cached and critic.fit_step==1921 and critic.pending_fit is None
    assert all(step==1921 and kw['log_step']==1920 for step,_,kw,_ in fits)


@pytest.mark.parametrize('failure', ['outside', 'invalid_negative', 'fit_error'])
def test_rejection_and_fit_failure_restore_all_adamw_state_parameters_buffers_and_scheduler(tmp_path,failure):
    actor,opt,scheduler,critic,guard,fits=fixture(tmp_path)
    weights=copy.deepcopy(actor.state_dict()); moments=copy.deepcopy(opt.state_dict())
    clock=copy.deepcopy(scheduler.state_dict()); old_head=critic.source.head
    if failure=='fit_error':
        def fail(*args,**kwargs):
            actor.normalizer.add_(7)
            raise ValueError('undertrained candidate fixture')
        critic.fit_candidate=fail
        with pytest.raises(ValueError,match='undertrained'):
            guard.propose(actor,opt,1920)
    else:
        value=.02 if failure=='outside' else -.1
        guard._evaluate=lambda *args:dict(mean=value,upper=value,standard_error=0,accepted=False)
        accepted,metrics=guard.propose(actor,opt,1920)
        assert not accepted and metrics['_tau_classifier_trust_stop'] and len(fits)==5
    for k,v in weights.items():torch.testing.assert_close(actor.state_dict()[k],v,rtol=0,atol=0)
    assert_state_equal(opt.state_dict(),moments)
    assert scheduler.state_dict()==clock and critic.source.head is old_head
    assert critic.fit_step==1920 and critic.pending_fit is None
    assert guard.report['optimizer_state_restored']


def test_actual_validation_gate_uses_full_disjoint_current_draws_without_truth(tmp_path):
    actor,opt,scheduler,critic,guard,_=fixture(tmp_path)
    n=119002
    panel=dict(source_ids=np.arange(416701,416701+n),condition=np.zeros((n,2),dtype=np.float32),
        candidate_truth=np.full((n,3),np.nan,dtype=np.float32),event_weight=np.ones(n))
    critic.cycle.validation=panel
    critic.cycle.cfg['fit']=dict(batch_size=4096)
    features=np.full((n,3),.003,dtype=np.float32)
    calls=[]
    def generate(p,folder,k,seed,physics):
        assert p is panel and k==1 and not physics and seed!=42+1920
        calls.append(seed)
        torch.rand(9)  # generation's RNG and eval mode must be restored
        return dict(features=features)
    critic.cycle.generate=generate
    class Head(torch.nn.Module):
        def forward(self,c,f):
            assert torch.isfinite(f).all() and len(f)<=4096
            return f[:,0]
    head=Head().eval().requires_grad_(False)
    rng=torch.get_rng_state().clone(); mode=actor.training
    result=guard._evaluate((head,None,None),tmp_path,1920)
    assert result['accepted'] and result['mean']==pytest.approx(.003)
    assert result['effective_events']==pytest.approx(n) and len(calls)==1
    assert actor.training==mode and torch.equal(torch.get_rng_state(),rng)
    critic.cycle.validation['source_ids'][0]=0
    with pytest.raises(ValueError,match='disjoint'):
        guard._evaluate((None,None,None),tmp_path,1920)


def test_production_terminal_branch_retains_accepted_clock_and_saves_restored_incumbent():
    # Execute the real terminal branch without Ray/remote data. Its early
    # return must run BEFORE the loop's accepted-update counter increment.
    tree=ast.parse((ROOT/'evenet_dgpo/RL/DGPO_neutrino/dgpo_trainer.py').read_text())
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.If)
              and "'_tau_classifier_trust_stop'" in ast.unparse(n.test))
    events=[]
    ns=dict(metrics={'_tau_classifier_trust_stop':True},wandb_mod=object(),is_rank0=True,
        model=None,ema_save=None,optimizer=None,ref_model=None,epoch=192,steps_this_epoch=3,
        ema_rollout=None,round_ref_model=None,tau_source=SimpleNamespace(round_id=21),reward_checkpoint_metadata={},
        tau_mechanism_driver=SimpleNamespace(observer=SimpleNamespace(stop_at_trust_boundary=lambda step,m:events.append(('audit',step)))),
        _wandb_train_payload=lambda m:m,_wandb_log_auxiliary=lambda *a,**k:events.append(('log',k['current_global_step'])),
        _dgpo_save_last_ckpt=lambda *a,**k:events.append(('save',k['global_step'],k['dgpo_epoch_step'])),
        constraint_ckpt_payload_for_save=lambda:None,_adaptive_state_payload=lambda:None,
        _adaptive_stack_payload=lambda:None,_barrier=lambda:None,
        _finish_wandb_run=lambda *a:events.append(('finish',)),wandb_active=True,
        _log=SimpleNamespace(info=lambda *a:None))
    function=ast.parse('def terminal(global_step):\n    pass').body[0]
    function.body=[node]+ast.parse('global_step += 1\nreturn global_step').body
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),'terminal','exec'),ns)
    assert ns['terminal'](1923) is None
    assert events==[('log',1923),('audit',1923),('save',1923,3),('finish',)]


@pytest.mark.parametrize('passes', [True,False])
def test_real_trainer_optimizer_and_scheduler_blocks_tick_only_for_accepted_proposal(tmp_path,passes):
    actor,opt,scheduler,critic,guard,_=fixture(tmp_path)
    guard._evaluate=lambda *a:dict(mean=.001 if passes else .1,upper=.001 if passes else .1,
                                  standard_error=0,accepted=passes)
    wrapper=SimpleNamespace(optimizer=opt,step=opt.step,param_groups=opt.param_groups,
                            zero_grad=opt.zero_grad,scheduler_step=scheduler.step)
    tree=ast.parse((ROOT/'evenet_dgpo/RL/DGPO_neutrino/dgpo_trainer.py').read_text())
    train=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train_step')
    transaction=next(n for n in ast.walk(train) if isinstance(n,ast.Try)
                     and 'classifier_trust.propose' in ast.unparse(n))
    clock=next(n for n in ast.walk(train) if isinstance(n,ast.If)
               and ast.unparse(n.test)=='optimizer_ran or not trust_boundary_hit')
    namespace=dict(classifier_trust=guard,model=actor,optimizer=wrapper,global_step=1920,
        original_group_lrs=[g['lr'] for g in opt.param_groups],optimizer_ran=False,
        trust_boundary_hit=False,classifier_trust_metrics={},trust_optimizer_state_advanced=False)
    before=scheduler.last_epoch
    exec(compile(ast.fix_missing_locations(ast.Module(body=[transaction,clock],type_ignores=[])),
                 'actual_trainer_transaction','exec'),namespace)
    assert namespace['optimizer_ran'] is passes
    assert namespace['trust_boundary_hit'] is not passes
    assert scheduler.last_epoch==before+int(passes)
    assert guard.accepted_step==1920+int(passes)


def test_decision_publication_failure_is_an_atomic_rollback(tmp_path):
    actor,opt,scheduler,critic,guard,_=fixture(tmp_path)
    before=copy.deepcopy(actor.state_dict());moments=copy.deepcopy(opt.state_dict())
    guard._evaluate=lambda *a:dict(mean=.001,upper=.001,standard_error=0,accepted=True)
    def fail(*a):raise OSError('decision output fixture')
    guard._publish=fail
    with pytest.raises(OSError,match='decision output'):
        guard.propose(actor,opt,1920)
    for k,v in before.items():torch.testing.assert_close(actor.state_dict()[k],v,rtol=0,atol=0)
    assert_state_equal(opt.state_dict(),moments)
    assert guard.accepted_step==1920 and critic.pending_fit is None


def test_terminal_audit_adds_endpoint_without_completing_predeclared_budget(tmp_path):
    probe=object.__new__(TauRewardTransferProbe)
    probe.source_step=1920;probe.last_relative_step=3
    probe.cfg=dict(audit_relative_steps=[0,50]);probe.results={};probe.output=tmp_path
    emitted=[]
    probe.cycle=SimpleNamespace(rank=0,world=1,emit=lambda *a:emitted.append(a))
    probe._assert_frozen=lambda:None
    audits=[]
    probe._measure=lambda step,*a:audits.append((step,list(probe.cfg['audit_relative_steps'])))
    probe.stop_at_trust_boundary(1923,{})
    assert audits==[(3,[0,3,50])] and probe.cfg['audit_relative_steps']==[0,50]
    assert probe.results['complete'] is False and probe.results['accepted_updates']==3
    assert (tmp_path/'report.json').is_file() and emitted[-1][1]==1923


def test_new_real_config_preserves_soft_objective_and_16gpu_filtered_source_contract(tmp_path):
    base=yaml.safe_load((ROOT/'config/tau_full_trajectory_1920_mb64_classifier_kl.yaml').read_text())
    trial=yaml.safe_load((ROOT/'config/tau_full_trajectory_1920_mb64_classifier_kl_trust001.yaml').read_text())
    stripped=copy.deepcopy(trial)
    assert stripped['full_trajectory'].pop('hard_trust_region')==TRUST
    stripped['series_root']=base['series_root']
    assert stripped==base
    cfg=launch.configure(source_runtime(),trial,launch.source_metadata(saved_state()),
        pinned=tmp_path/'source.ckpt',output=tmp_path,stage='trajectory',method='pathwise')
    assert cfg['dgpo']['reference_trust']['coefficient']==0
    assert cfg['experiment']['classifier_kl']==OPTIONS and cfg['experiment']['hard_trust_region']==TRUST
    assert cfg['dgpo']['tau_ratio']['mechanism_probe']['full_trajectory']['reward_refit']==REFIT
    assert cfg['platform']['number_of_workers']==16
    assert cfg['logger']['wandb']['fresh_run'] and 'KL<=0.01' in cfg['logger']['wandb']['name']
