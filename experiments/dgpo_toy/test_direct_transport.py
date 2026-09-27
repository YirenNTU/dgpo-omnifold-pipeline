import copy
from dataclasses import replace
import torch
from experiments.dgpo_toy.conditional import Config,Distribution,Denoiser,Classifier,generator,policy_train
from experiments.dgpo_toy.direct_transport import FixedTransport,TransportReward,BaseCoordinates,physical_panel
from experiments.dgpo_toy.low_ess_recovery import fresh_batch
from experiments.dgpo_toy.test_fixed_reward import assert_nested_equal


def test_transform_batch_shapes_and_reward_match():
    cfg=replace(Config(),dimensions=6,hidden=8,classifier_hidden=8)
    d=Distribution(cfg);t=FixedTransport(d);r=generator(92)
    c=d.contexts(12,r)[:,None].expand(-1,8,-1);x=torch.randn(12,8,6,generator=r)
    identity=FixedTransport(d,0.)
    assert identity(x,c) is x
    y=t(x,c);assert y.shape==x.shape and torch.isfinite(y).all()
    f=Classifier(cfg,True).eval();wrapped=TransportReward(f,t)
    torch.testing.assert_close(wrapped(x,c,d),f(y,c,d))
    torch.testing.assert_close(BaseCoordinates(cfg,t).joint_signal(x,c),d.joint_signal(y,c))


def test_fit_transform_only_changes_negatives():
    cfg=replace(Config(),dimensions=6,hidden=8,fit_batch=32,ddim_steps=2)
    d=Distribution(cfg);m=Denoiser(cfg)
    y,c,l=fresh_batch(m,d,cfg,generator(8))
    yy,cc,ll=fresh_batch(m,d,cfg,generator(8),lambda x,c:x+3)
    assert torch.equal(y[:16],yy[:16]) and torch.equal(c,cc) and torch.equal(l,ll)
    torch.testing.assert_close(yy[16:],y[16:]+3)


def test_identity_transport_exact_native_updates():
    torch.set_num_threads(1)
    cfg=replace(Config(),dimensions=6,hidden=8,classifier_hidden=8,policy_steps=2,eval_events=8,batch=4,candidates=3,ddim_steps=2,timesteps=2)
    d=Distribution(cfg);m=Denoiser(cfg);f=Classifier(cfg,True).eval().requires_grad_(False)
    snapshots=[]
    for wrapped in (f,TransportReward(f,FixedTransport(d,0.))):
        states=[]
        def save(step,model,opt,rng,history):
            states.append(copy.deepcopy({"model":model.state_dict(),"optimizer":opt.state_dict(),"rng":rng.get_state()}))
        policy_train('dgpo',m,wrapped,d,cfg,17,90017,lambda r:None,save,velocity_coefficient=1.)
        snapshots.append(states)
    assert_nested_equal(snapshots[0],snapshots[1])
