from dataclasses import asdict,replace
import pytest
import torch
from experiments.dgpo_toy.conditional import Config,Denoiser,generator,dgpo_objective
from experiments.dgpo_toy.parity_cube import CubeDistribution,ModeReward
from experiments.dgpo_toy.cube_lockdown import (ConditionDenoiser,condition_features,
    verify_initial,contrast,conclusions)


def small_config():
    return Config(dimensions=3,context_dim=1,hidden=8,batch=2,candidates=4,timesteps=2,ddim_steps=2)


def test_initial_function_and_parameter_count_exact():
    torch.set_num_threads(1);cfg=small_config();base=Denoiser(cfg)
    variants={b:ConditionDenoiser(cfg,b,base.state_dict()) for b in ('raw','polynomial','fourier')}
    assert all(v['samples_exact'] for v in verify_initial(base,variants,cfg).values())


def test_feature_scale_matches_without_fitting_to_evaluation():
    c=((torch.arange(65536,dtype=torch.float64)+.5)/65536*2-1)[:,None]
    for basis in ('polynomial','fourier'):
        x=condition_features(c,basis)
        torch.testing.assert_close(x.mean(0),torch.zeros(8,dtype=x.dtype),atol=1e-7,rtol=0)
        torch.testing.assert_close(x.T@x/len(c),torch.eye(8,dtype=x.dtype),atol=1e-6,rtol=0)


def test_zero_adapter_can_learn_and_raw_preserves_original_update():
    cfg=small_config();base=Denoiser(cfg)
    data=CubeDistribution(cfg,continuous=True,reference_sharpness=4.,condition_frequency=8)
    rng=generator(4);c=data.contexts(2,rng);z=torch.randn(2,4,3,generator=rng)
    t=torch.rand(2,2,generator=rng)*.7;eps=torch.randn(2,2,3,generator=rng)
    variants=[base]+[ConditionDenoiser(cfg,b,base.state_dict()) for b in ('raw','polynomial','fourier')]
    for m in variants:
        loss,_=dgpo_objective(m,m,ModeReward(),data,cfg,c,z,t,eps,velocity_coefficient=1.)
        loss.backward()
    for (_,p),(_,q) in zip(base.named_parameters(),variants[1].named_parameters()):
        torch.testing.assert_close(p.grad,q.grad,atol=0,rtol=0)
    assert variants[1].condition_adapter.weight.grad.abs().sum()==0
    assert variants[2].condition_adapter.weight.grad.abs().sum()>0
    assert variants[3].condition_adapter.weight.grad.abs().sum()>0


def test_contrast_uses_context_not_candidates_as_independent_units():
    a=torch.arange(32,dtype=torch.float32)[:,None]
    one=contrast(a,torch.zeros_like(a));many=contrast(a.expand(-1,8),torch.zeros(32,8))
    assert one==many
    assert conclusions({'seeds':[17],'arms':{}})['state']=='pending_all_prespecified_arms'


def test_six_arm_tiny_model_smoke(tmp_path,monkeypatch):
    from experiments.dgpo_toy import cube_lockdown as runner
    torch.set_num_threads(1);cfg=small_config();source=tmp_path/'source';source.mkdir()
    torch.save({'model':Denoiser(cfg).state_dict(),'config':asdict(cfg),
        'condition_frequency':8,'step':1},source/'best_pretrain.pt')
    original_evaluate=runner.evaluate
    monkeypatch.setattr(runner,'evaluate',lambda m,d,c,s:original_evaluate(m,d,replace(c,eval_events=256),s))
    result=runner.run(source,tmp_path/'out',steps=300,seeds=(17,),wandb_mode='disabled')
    assert result['state']=='completed' and len(result['arms'])==6
    assert all(a['completed_steps']==300 and a['source_unchanged'] for a in result['arms'].values())
    assert result['decision']['state']=='completed_prespecified_comparisons'
    assert len(result['contrasts']['17']['300'])==4
