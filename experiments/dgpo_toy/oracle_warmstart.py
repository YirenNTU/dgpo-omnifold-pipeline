"""Oracle moments ONLY during warm-start; frozen learned H4 and native DGPO after."""
import argparse
import copy
import json
import math
from dataclasses import asdict, replace
from pathlib import Path
import torch
from .conditional import Distribution, generator, ddim, make_panel, classifier_metrics, policy_panel, paired_gain, policy_train
from .fixed_reward import load_source, unchanged
from .truth_pretrain import atomic_json, atomic_checkpoint, noisy_target, sample_loss
from .structure_metrics import structure_features, structure_target, structure_panel, paired_structure_change
from .broad_coverage import region_hits, coverage
from .low_ess_recovery import fit_streaming


@torch.no_grad()
def sample_teacher(c,noise,data,amplitude=.6):
    """Triangular copula transport, conditional density 1+a*cos(triplephase).

    First two coordinates stay unchanged; invert the third conditional CDF.
    All univariate/pair marginals remain Gaussian in the ideal continuous map.
    Four independent triple groups; unlike truth's shared mixture membership.
    """
    if not 0 <= amplitude < 1:
        raise ValueError("Teacher amplitude must lie in [0,1)")
    if amplitude==0:
        return data.mean(c)+noise
    z=noise.double().reshape(len(noise),-1,3)
    u=(.5*(1+torch.erf(z/2**.5))).clamp(1e-12,1-1e-12)
    offset=2*torch.pi*u[...,:2].sum(-1)-data.phase(c).double()
    lo=torch.zeros_like(offset);hi=torch.ones_like(offset)
    for _ in range(45):
        mid=(lo+hi)/2
        cdf=mid+amplitude/(2*torch.pi)*(torch.sin(2*torch.pi*mid+offset)-torch.sin(offset))
        below=cdf<u[...,2]
        lo=torch.where(below,mid,lo);hi=torch.where(below,hi,mid)
    out=z.clone()
    out[...,2]=2**.5*torch.erfinv(2*((lo+hi)/2).clamp(1e-12,1-1e-12)-1)
    return data.mean(c)+out.flatten(1).to(noise.dtype)


def sample_teacher_loss(model,c,noise,data):
    target=sample_teacher(c,noise,data)
    output=ddim(model,c,noise,data.cfg.ddim_steps)
    loss=(output-target).square().mean()
    return loss,{"teacher_mse":float(loss.detach())}


def oracle_loss(model,c,ytruth,noise,data,rng,target):
    """Full pathwise oracle warm-up, never called by policy_train."""
    y=ddim(model,c,noise,data.cfg.ddim_steps)
    z=y-data.mean(c)
    moments=structure_features(y,c,data).mean(0)
    joint=(moments-target).square().sum()
    centered=z-z.mean(0)
    covariance=centered.T@centered/len(z)
    low=z.mean(0).square().sum()+(covariance-torch.eye(z.shape[-1])).square().sum()
    # Match conditional Gaussian univariate ranks without clipping outputs.
    sorted_z=z.sort(0).values
    probs=(torch.arange(len(z),dtype=z.dtype)+.5)/len(z)
    normal=2**.5*torch.erfinv(2*probs-1)
    marginal=(sorted_z-normal[:,None]).square().mean()
    x,t,v=noisy_target(ytruth,rng)
    ordinary=sample_loss(model,(c,x,t,v))
    loss=joint+5*low+10*marginal+.1*ordinary
    return loss,{"joint_loss":float(joint.detach()),"low_loss":float(low.detach()),
                 "marginal_loss":float(marginal.detach()),"ordinary_loss":float(ordinary.detach())}


@torch.no_grad()
def probe(model,data,cfg,seed,n=4096):
    rng=generator(seed);c=data.contexts(n,rng)
    noise=torch.randn(n,cfg.dimensions,generator=rng)
    y=torch.cat([ddim(model,cc,z,cfg.ddim_steps) for cc,z in zip(c.split(512),noise.split(512))])
    z=y-data.mean(c);zc=z-z.mean(0);cov=zc.T@zc/len(z)
    return {"joint_mean":float(data.joint_signal(y,c).mean()),
        "hit_fraction":float(region_hits(y,c,data).float().mean()),
        "mean_absmax":float(z.mean(0).abs().max()),
        "variance_error":float((cov.diag()-1).abs().max()),
        "pair_covariance":float((cov-torch.diag(cov.diag())).abs().max())}


def candidate_gate(m,base):
    return (.10<m["joint_mean"]<.50 and m["hit_fraction"]>2*base["hit_fraction"]
        and m["mean_absmax"]<.1 and m["variance_error"]<.15 and m["pair_covariance"]<.08)


def run(root,output,warm_steps=3000,policy_steps=10000,warm_mode="moments",coefficient=1.):
    if warm_steps<1 or policy_steps<1:raise ValueError("Positive budgets required")
    if warm_mode not in ("moments","sample_teacher"):raise ValueError("Unknown warm mode")
    if not math.isfinite(coefficient) or coefficient<0:raise ValueError("Finite nonnegative coefficient required")
    cfg,original,_,source,_=load_source(root/"low_ess_joint3_v1/reward.pt")
    cfg=replace(cfg,policy_steps=policy_steps)
    data=Distribution(cfg)
    output.mkdir(parents=True,exist_ok=False)
    report={"state":"warming","source":str((root/"low_ess_joint3_v1/reward.pt").resolve()),
        "config":asdict(cfg),"warm_budget":warm_steps,"policy_budget":policy_steps,
        "velocity_coefficient":coefficient,"warm_mode":warm_mode,
        "oracle_target_fraction":.3 if warm_mode=="moments" else None,
        "teacher_amplitude":.6 if warm_mode=="sample_teacher" else None,
        "scope":"Oracle auxiliary used only for initialization; not an ESS-only intervention; raw weights.",
        "selection_seed":410017,"confirmation_seed":420017,"warm_history":[]}
    atomic_json(output/"report.json",report)
    model=copy.deepcopy(original).requires_grad_(True)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=cfg.weight_decay)
    rng=generator(400017)
    target=torch.tensor(structure_target(data),dtype=torch.float32)*.3
    baseline=probe(original,data,cfg,410017)
    report["selection_baseline"]=baseline
    if warm_mode=="sample_teacher":
        teacher_rng=generator(415017)
        cc=data.contexts(32768,teacher_rng)
        zz=torch.randn(32768,cfg.dimensions,generator=teacher_rng)
        teacher=sample_teacher(cc,zz,data)
        residual=teacher-data.mean(cc);centered=residual-residual.mean(0)
        cov=centered.T@centered/len(residual)
        report["teacher_validation"]={"joint_mean":float(data.joint_signal(teacher,cc).mean()),
            "hit_fraction":float(region_hits(teacher,cc,data).float().mean()),
            "mean_absmax":float(residual.mean(0).abs().max()),
            "variance_error":float((cov.diag()-1).abs().max()),
            "pair_covariance":float((cov-torch.diag(cov.diag())).abs().max())}
        if not candidate_gate(report["teacher_validation"],baseline):
            report["state"]="teacher_gate_failed";atomic_json(output/"report.json",report);return report
        atomic_json(output/"report.json",report)
    selected=False
    try:
        for step in range(1,warm_steps+1):
            c=data.contexts(512,rng)
            truth=data.sample(c,rng,truth=True) if warm_mode=="moments" else None
            noise=torch.randn(512,cfg.dimensions,generator=rng)
            loss,diag=(oracle_loss(model,c,truth,noise,data,rng,target) if warm_mode=="moments"
                       else sample_teacher_loss(model,c,noise,data))
            if not torch.isfinite(loss):raise FloatingPointError("Nonfinite oracle loss")
            opt.zero_grad(set_to_none=True);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
            if step%100==0 or step==warm_steps:
                m=probe(model,data,cfg,410017)
                row={"step":step,"loss":float(loss.detach()),"gradient_norm":float(norm),**diag,**m}
                report["warm_history"].append(row);print(json.dumps(row),flush=True)
                selected=candidate_gate(m,baseline)
                atomic_checkpoint(output/"warm_state.pt",{"model":model.state_dict(),"optimizer":opt.state_dict(),
                    "rng":rng.get_state(),"step":step,"config":asdict(cfg),"oracle_assisted":True})
                atomic_json(output/"report.json",report)
                if selected:break
        report["warm_step"]=step
        initial=model.eval().requires_grad_(False)
        base_cov=coverage(original,data,cfg,420017);new_cov=coverage(initial,data,cfg,420017)
        report["coverage"]={"initial":{k:float(v.mean()) for k,v in base_cov.items()},
            "warm":{k:float(v.mean()) for k,v in new_cov.items()}}
        report["coverage_change"]={k:paired_gain(new_cov[k][:,None],base_cov[k][:,None]) for k in base_cov}
        confirm=probe(initial,data,cfg,420018,16384);base_confirm=probe(original,data,cfg,420018,16384)
        report["confirmation"]=confirm
        passed=selected and candidate_gate(confirm,base_confirm) and report["coverage_change"]["any_k8"]["lo95"]>0
        report["warm_gate"]=passed
        if not passed:
            report["state"]="warm_gate_failed";return report
        atomic_checkpoint(output/"warm_model.pt",{"model":initial.state_dict(),"config":asdict(cfg),"step":step,"weights":"raw","oracle_assisted":True})
        report["state"]="fitting_strong";atomic_json(output/"report.json",report)
        panels={name:make_panel(initial,data,cfg,count,17+offset) for name,count,offset in
                (("train",cfg.train_events,10000),("validation",cfg.validation_events,20000),("test",cfg.test_events,30000))}
        with (output/"progress.jsonl").open("w") as log:
            def emit(row):
                log.write(json.dumps(row,allow_nan=False)+"\n");log.flush()
                print(json.dumps(row,allow_nan=False),flush=True)
            critic,fit=fit_streaming(initial,data,cfg,panels["train"],panels["validation"],17,emit,feature_mode="joint3")
            report["classifier_fit"]=fit
            report["classifier_test"]=classifier_metrics(critic,panels["test"],data)
            atomic_checkpoint(output/"classifier.pt",{"classifier":critic.state_dict(),"config":asdict(cfg),"fit":fit})
            initial_state=copy.deepcopy(initial.state_dict());critic_state=copy.deepcopy(critic.state_dict())
            report["state"]="training_dgpo";atomic_json(output/"report.json",report)
            def checkpoint(step,m,optimizer,random,history):
                if step%250==0:
                    _,summary=structure_panel(m,data,cfg,90017)
                    emit({"phase":"structure","step":step,**summary})
                if step==1 or step%1000==0 or step==policy_steps:
                    atomic_checkpoint(output/"dgpo_state.pt",{"model":m.state_dict(),"optimizer":optimizer.state_dict(),
                        "rng":random.get_state(),"history":history,"step":step,"config":asdict(cfg),
                        "velocity_coefficient":coefficient,"monitor_seed":90017,"seed":17,"endpoint_controller":None})
            final,history=policy_train("dgpo",initial,critic,data,cfg,17,90017,emit,checkpoint,velocity_coefficient=coefficient)
        report["endpoints"]={};rewards={};features={}
        for name,m in (("original",original),("warm",initial),("dgpo",final)):
            rewards[name],_=policy_panel(m,critic,data,cfg,440017)
            features[name],report["endpoints"][name]=structure_panel(m,data,cfg,440017)
        report["reward_gain"]=paired_gain(rewards["dgpo"],rewards["warm"])
        report["structure_change"]=paired_structure_change(features["dgpo"],features["warm"],structure_target(data),440018)
        report["sources_unchanged"]=unchanged(initial,initial_state) and unchanged(critic,critic_state) and unchanged(original,source["initial"])
        report.update(state="completed",completed_policy_steps=len(history))
    except BaseException as exc:
        report.update(state="failed_or_interrupted",error=repr(exc));raise
    finally:atomic_json(output/"report.json",report)
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("artifacts/dgpo_toy"))
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--warm-steps",type=int,default=3000)
    p.add_argument("--policy-steps",type=int,default=10000)
    p.add_argument("--warm-mode",choices=("moments","sample_teacher"),default="moments")
    p.add_argument("--coefficient",type=float,default=1.)
    args=p.parse_args();torch.set_num_threads(1);run(**vars(args))
