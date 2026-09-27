import copy
from dataclasses import replace, asdict
import torch
from experiments.dgpo_toy.conditional import Config, Distribution, Denoiser, Classifier, dgpo_objective
from experiments.dgpo_toy.direct_transport import FixedTransport
from experiments.dgpo_toy.coverage_budget import draw_pool, loss_inputs, accumulate, ObservedReward
from experiments.dgpo_toy.coverage_budget import ARMS, restore_arms


def setup():
    torch.set_num_threads(1)
    torch.manual_seed(51)
    cfg=replace(Config(),dimensions=6,hidden=8,classifier_hidden=8,batch=3,timesteps=2,ddim_steps=2)
    data=Distribution(cfg);model=Denoiser(cfg)
    ref=copy.deepcopy(model).requires_grad_(False)
    critic=Classifier(cfg,True).eval().requires_grad_(False)
    reward=ObservedReward(critic,FixedTransport(data,0.))
    return cfg,data,model,ref,reward,draw_pool(data,cfg,81)


def test_shared_pool_and_budget():
    cfg,data,m,ref,reward,pool=setup()
    a=list(loss_inputs(pool,"A_k8"));b=list(loss_inputs(pool,"B_k128"));c=list(loss_inputs(pool,"C_16xk8"))
    assert len(a)==len(b)==1 and len(c)==16
    assert torch.equal(a[0][1],b[0][1][:,:8])
    assert torch.equal(torch.cat([x[1] for x in c],1),b[0][1])
    assert all(torch.equal(x[0],b[0][0]) for x in c)
    assert sum(x[1].numel() for x in c)==b[0][1].numel()


def test_accumulation_matches_mean_of_complete_losses_and_components():
    cfg,data,m,ref,reward,pool=setup()
    # Nonzero reference difference checks the penalty and nonlinear gate too.
    with torch.no_grad():
        for p in m.parameters():p.add_(.002)
    expected=copy.deepcopy(m)
    losses=[dgpo_objective(expected,ref,reward,data,cfg,*inputs,velocity_coefficient=1.)[0]
            for inputs in loss_inputs(pool,"C_16xk8")]
    torch.stack(losses).mean().backward()
    row,vec=accumulate(m,ref,reward,data,cfg,pool,"C_16xk8",True)
    for p,q in zip(m.parameters(),expected.parameters()):
        torch.testing.assert_close(p.grad,q.grad,rtol=2e-5,atol=1e-8)
    actual=torch.cat([p.grad.flatten().double() for p in m.parameters()])
    torch.testing.assert_close(actual,vec["total"],rtol=2e-5,atol=1e-8)
    assert row["loss_calls"]==16
    assert row["candidates_per_update"]==cfg.batch*128
    assert row["update_pool_hit_fraction"]>=row["loss_group_hit_fraction"]


def test_a_matches_native_loss_gradient():
    cfg,data,m,ref,reward,pool=setup()
    expected=copy.deepcopy(m)
    loss,_=dgpo_objective(expected,ref,reward,data,cfg,*next(loss_inputs(pool,"A_k8")),velocity_coefficient=1.)
    loss.backward()
    accumulate(m,ref,reward,data,cfg,pool,"A_k8")
    for p,q in zip(m.parameters(),expected.parameters()):
        torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)


def test_resume_preserves_adamw_and_next_update(tmp_path):
    cfg,data,m,ref,reward,pool=setup()
    models={a:copy.deepcopy(m) for a in ARMS}
    opts={a:torch.optim.AdamW(v.parameters(),lr=cfg.policy_lr,weight_decay=cfg.weight_decay) for a,v in models.items()}
    for a,v in models.items():
        accumulate(v,ref,reward,data,cfg,pool,a)
        torch.nn.utils.clip_grad_norm_(v.parameters(),1.)
        opts[a].step()
        torch.save(dict(model=v.state_dict(),optimizer=opts[a].state_dict(),step=1,arm=a,
            config=asdict(cfg),source=str(tmp_path/"reward.pt"),seed=17,amplitude=0.,velocity_coefficient=1.),tmp_path/(a+".pt"))
    restored={a:copy.deepcopy(m) for a in ARMS}
    optimizers={a:torch.optim.AdamW(v.parameters(),lr=1.) for a,v in restored.items()}
    assert restore_arms(tmp_path,restored,optimizers,cfg,tmp_path/"reward.pt",0.,17)==1
    for a in ARMS:
        for model,opt in ((models[a],opts[a]),(restored[a],optimizers[a])):
            opt.zero_grad(set_to_none=True)
            accumulate(model,ref,reward,data,cfg,draw_pool(data,cfg,82),a)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
        for p,q in zip(models[a].parameters(),restored[a].parameters()):
            torch.testing.assert_close(p,q,rtol=0,atol=0)
