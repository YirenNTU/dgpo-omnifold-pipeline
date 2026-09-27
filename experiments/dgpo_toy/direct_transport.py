"""Explicit model cheat: fixed physical transport after a trainable neural DDIM.

Native DGPO stays in base diffusion coordinates with reward f(T(x),c).
No truth injection, no oracle reward, no pathwise transport gradient in RL.
"""
import argparse
import copy
import json
import math
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
import torch
from .conditional import Distribution, generator, ddim, make_panel, classifier_metrics, policy_train, paired_gain, weight_health
from .fixed_reward import load_source, unchanged
from .oracle_warmstart import sample_teacher
from .low_ess_recovery import fit_streaming
from .structure_metrics import structure_features, structure_target, paired_structure_change
from .broad_coverage import region_hits
from .truth_pretrain import atomic_json, atomic_checkpoint


class FixedTransport:
    def __init__(self,data,amplitude=.6):
        if not math.isfinite(amplitude) or not 0<=amplitude<1:
            raise ValueError("Finite amplitude in [0,1) required")
        self.data,self.amplitude=data,amplitude

    @torch.no_grad()
    def __call__(self,x,c):
        if self.amplitude==0:return x
        shape=x.shape
        cc=c.expand(*shape[:-1],c.shape[-1]).reshape(-1,c.shape[-1])
        xx=x.reshape(-1,shape[-1])
        return sample_teacher(cc,xx-self.data.mean(cc),self.data,self.amplitude).reshape(shape)


class TransportReward(torch.nn.Module):
    def __init__(self,critic,transport):
        super().__init__();self.critic=critic;self.transport=transport

    def forward(self,x,c,data):
        return self.critic(self.transport(x,c),c,self.transport.data)


class BaseCoordinates(Distribution):
    """Conditions unchanged; built-in joint monitor explicitly evaluates physical y."""
    def __init__(self,cfg,transport):
        super().__init__(cfg);self.transport=transport

    def joint_signal(self,x,c):
        return self.transport.data.joint_signal(self.transport(x,c),c)


@torch.no_grad()
def physical_panel(model,critic,transport,cfg,seed,k=None):
    k=cfg.candidates if k is None else k
    data=transport.data;rng=generator(seed)
    c=data.contexts(cfg.eval_events,rng)
    noise=torch.randn(cfg.eval_events,k,cfg.dimensions,generator=rng)
    rewards=[];features=[];residuals=[];hits=[];saturated=[]
    for cc,z in zip(c.split(64),noise.split(64)):
        context=cc[:,None].expand(-1,k,-1)
        x=ddim(model,cc[:,None],z,cfg.ddim_steps)
        y=transport(x,context)
        if not torch.isfinite(y).all():raise FloatingPointError("Nonfinite physical output")
        if critic is not None:rewards.append(critic(y,context,data))
        features.append(structure_features(y,context,data).mean(1))
        residuals.append((y-data.mean(context)).flatten(0,1))
        hits.append(region_hits(y,context,data).float())
        saturated.append(((x-data.mean(context)).abs()>7).float().flatten())
    f=torch.cat(features).double().numpy();z=torch.cat(residuals).double();h=torch.cat(hits)
    centered=z-z.mean(0);cov=centered.T@centered/len(z)
    summary={"moment_rmse":float(np.sqrt(np.mean((f.mean(0)-structure_target(data))**2))),
        "mean_absmax":float(z.mean(0).abs().max()),"variance_error":float((cov.diag()-1).abs().max()),
        "pair_covariance":float((cov-torch.diag(cov.diag())).abs().max()),
        "joint_mean":float(f[:,:cfg.dimensions//3*4:4].mean()),
        "per_candidate_hit":float(h.mean()),"any_k_hit":float(h.amax(1).mean()),
        "base_residual_abs_gt7_fraction":float(torch.cat(saturated).mean()),"k":k}
    reward=torch.cat(rewards) if rewards else None
    if reward is not None:
        summary.update(reward_mean=float(reward.mean()),reward_std=float(reward.std(unbiased=False)),
            centered_reward_rms=float((reward-reward.mean(1,keepdim=True)).square().mean().sqrt()),
            within_k_ess=float((1/(k*reward.softmax(1).square().sum(1))).mean()),
            **weight_health(reward))
    return reward,f,h,summary


def run(root,output,steps=10000,amplitude=.6):
    if steps<1:raise ValueError("Positive policy budget required")
    cfg,initial,old_critic,payload,evidence=load_source(root/"low_ess_joint3_v1/reward.pt")
    cfg=replace(cfg,policy_steps=steps)
    data=Distribution(cfg);transport=FixedTransport(data,amplitude);identity=FixedTransport(data,0.)
    base_data=BaseCoordinates(cfg,transport)
    output.mkdir(parents=True,exist_ok=False)
    report={"state":"checking_direct_model","source":str((root/"low_ess_joint3_v1/reward.pt").resolve()),
        "config":asdict(cfg),"amplitude":amplitude,"velocity_coefficient":1.,"new_warm_updates":0,
        "model":"y=T_a(DDIM_v(c,noise),c); T fixed during fitting, DGPO and evaluation",
        "velocity_coordinates":"base diffusion x, not physical y; MSE is not exact KL",
        "seeds":{"train":3017,"monitor":90017,"coverage":450017,"endpoint":460017},
        "causal_scope":"Coverage/architecture intervention, not ESS-only causal identification.",
        "old_classifier_confirmation":evidence["confirmation"]}
    atomic_json(output/"report.json",report)
    try:
        _,_,before_hits,before=physical_panel(initial,None,identity,cfg,450017,k=128)
        _,_,after_hits,after=physical_panel(initial,None,transport,cfg,450017,k=128)
        change=paired_gain(after_hits[:,:8].amax(1)[:,None],before_hits[:,:8].amax(1)[:,None])
        report.update(initial_identity=before,initial_transformed=after,coverage_k8_change=change)
        passed=(change["lo95"]>0 and .1<after["joint_mean"]<.5 and after["mean_absmax"]<.1
            and after["variance_error"]<.15 and after["pair_covariance"]<.08)
        report["coverage_gate"]=passed
        if not passed:report["state"]="coverage_gate_failed";return report
        panels={}
        for name,n,offset in (("train",cfg.train_events,10000),("validation",cfg.validation_events,20000),("test",cfg.test_events,30000)):
            panel=make_panel(initial,data,cfg,n,17+offset)
            panel["generated"]=transport(panel["generated"],panel["context"])
            panels[name]=panel
        report["state"]="fitting_strong";atomic_json(output/"report.json",report)
        with (output/"progress.jsonl").open("w") as log:
            def emit(row):
                line=json.dumps(row,allow_nan=False);log.write(line+"\n");log.flush()
                if row.get("phase")!="policy" or row.get("step",0)%1000==0 or row.get("step")==1:
                    print(line,flush=True)
            critic,fit=fit_streaming(initial,data,cfg,panels["train"],panels["validation"],17,emit,
                feature_mode="joint3",sample_transform=transport)
            report["classifier_fit"]=fit;report["classifier_test"]=classifier_metrics(critic,panels["test"],data)
            wrapped=TransportReward(critic,transport).eval().requires_grad_(False)
            critic_state=copy.deepcopy(critic.state_dict())
            atomic_checkpoint(output/"reward.pt",{"config":asdict(cfg),"initial":initial.state_dict(),
                "classifier":critic_state,"amplitude":amplitude,"feature_mode":"joint3", "source":report["source"]})
            _,_,_,report["initial_signal"] =physical_panel(initial,critic,transport,cfg,90017)
            report["state"]="training_dgpo";atomic_json(output/"report.json",report)
            def checkpoint(step,model,opt,rng,history):
                if step==1 or step%250==0 or step==steps:
                    _,_,_,summary=physical_panel(model,critic,transport,cfg,90017)
                    emit({"phase":"physical_diagnostics","step":step,**summary})
                if step==1 or step%1000==0 or step==steps:
                    atomic_checkpoint(output/"dgpo_state.pt",{"model":model.state_dict(),"optimizer":opt.state_dict(),
                        "rng":rng.get_state(),"history":history,"step":step,"config":asdict(cfg),
                        "seed":17,"monitor_seed":90017,"velocity_coefficient":1.,"endpoint_controller":None,
                        "amplitude":amplitude,"source":report["source"],"model_coordinates":"base"})
            model,history=policy_train("dgpo",initial,wrapped,base_data,cfg,17,90017,emit,checkpoint,velocity_coefficient=1.)
        before_r,before_f,_,before=physical_panel(initial,critic,transport,cfg,460017)
        after_r,after_f,_,after=physical_panel(model,critic,transport,cfg,460017)
        report.update(state="completed",completed_steps=len(history),endpoint_initial=before,endpoint=after,
            own_reward_gain=paired_gain(after_r,before_r),
            structure_change=paired_structure_change(after_f,before_f,structure_target(data),460018),
            frozen_sources_unchanged=unchanged(initial,payload["initial"]) and unchanged(critic,critic_state))
        report["joint_and_low_order_pass"]=(report["structure_change"]["hi95"]<0 and after["mean_absmax"]<.1
            and after["variance_error"]<.15 and after["pair_covariance"]<.08)
    except BaseException as exc:
        report.update(state="failed_or_interrupted",error=repr(exc));raise
    finally:atomic_json(output/"report.json",report)
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("artifacts/dgpo_toy"))
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--steps",type=int,default=10000)
    p.add_argument("--amplitude",type=float,default=.6)
    args=p.parse_args();torch.set_num_threads(1);run(**vars(args))
