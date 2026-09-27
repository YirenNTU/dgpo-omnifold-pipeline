"""Conditional eight-corner diffusion; oracle mode reward, no classifier."""
import argparse
import copy
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import torch
from .conditional import Config, Denoiser, generator, ddim, policy_train, paired_gain
from .truth_pretrain import (atomic_json, atomic_checkpoint, make_panel,
    noisy_target, validation, early_stop_update)
from .coverage_budget import flatten_metrics


class CubeDistribution:
    def __init__(self, cfg, mass=.1, width=.15, continuous=False, reference_sharpness=1., condition_frequency=None):
        if cfg.dimensions!=3 or cfg.context_dim!=1:
            raise ValueError("Cube requires three outputs and one condition")
        if not 0<mass<1 or width<=0:
            raise ValueError("Invalid cube mixture")
        self.cfg,self.mass,self.width=cfg,mass,width
        self.continuous=continuous
        if condition_frequency is not None and (not continuous or isinstance(condition_frequency,bool) or int(condition_frequency)!=condition_frequency or condition_frequency<1):
            raise ValueError("Frequency requires continuous conditions and a positive integer")
        self.condition_frequency=condition_frequency
        self.condition_bins=8 if condition_frequency is None else 32*int(condition_frequency)
        if reference_sharpness<1 or not math.isfinite(reference_sharpness):raise ValueError("Invalid reference sharpness")
        self.reference_sharpness=reference_sharpness
        self.centers=torch.tensor([[x,y,z] for x in (-1.,1.) for y in (-1.,1.) for z in (-1.,1.)])

    def contexts(self,n,rng):
        if self.continuous:return 2*torch.rand(n,1,generator=rng)-1
        return (2*torch.randint(2,(n,1),generator=rng)-1).float()

    def sample(self,c,rng,*,truth=False):
        mass=.9 if truth else self.mass
        first=(2*torch.randint(2,(len(c),2),generator=rng)-1).float()
        if self.continuous:
            positive=self.positive_probability(c[:,0],truth=truth)
            desired=torch.where(torch.rand(len(c),generator=rng)<positive,1.,-1.)
        else:
            desired=torch.where(torch.rand(len(c),generator=rng)<mass,c[:,0],-c[:,0])
        last=desired*first.prod(-1)
        center=torch.cat((first,last[:,None]),-1)
        return center+self.width*torch.randn(center.shape,generator=rng)

    def joint_signal(self,y,c):
        signs=torch.where(y>=0,1.,-1.)
        desired=torch.where(self.condition_signal(c[...,0])>=0,1.,-1.)
        return (signs.prod(-1)==desired).float()

    def condition_signal(self,c):
        return c if self.condition_frequency is None else torch.sin(self.condition_frequency*math.pi*c)

    def probabilities(self,c,mass=None):
        reference=mass is None
        mass=self.mass if reference else mass
        if self.continuous:
            cc=torch.as_tensor(c)
            positive=self.positive_probability(cc) if reference else .5+(mass-.5)*self.condition_signal(cc)
            return torch.where(self.centers.prod(-1)>0,positive[...,None],1-positive[...,None])/4
        good=self.centers.prod(-1)==c
        return torch.where(good,mass/4,(1-mass)/4)

    def positive_probability(self,c,truth=False):
        p=.5+(.4 if truth else self.mass-.5)*self.condition_signal(c)
        if truth or self.reference_sharpness==1:return p
        return torch.sigmoid(self.reference_sharpness*torch.logit(p))


class ModeReward(torch.nn.Module):
    """Categorical ideal ratio reward; NOT an actual diffusion endpoint ratio."""
    def __init__(self,mass=.1,target=.9):
        super().__init__()
        self.good=math.log(target/mass)
        self.other=math.log((1-target)/(1-mass))

    def forward(self,y,c,data):
        if data.continuous:
            parity=torch.where(y>=0,1.,-1.).prod(-1)
            signal=data.condition_signal(c[...,0])*parity
            p=data.positive_probability(c[...,0])
            ref=torch.where(parity>0,p,1-p)
            return torch.log(.5+.4*signal)-ref.log()
        return torch.where(data.joint_signal(y,c).bool(),self.good,self.other)


def cube_metrics(y,c,data):
    y=y.reshape(-1,3).double();c=c.reshape(-1)
    if len(c)!=len(y):raise ValueError("One condition per sample required")
    signs=torch.where(y>=0,1.,-1.)
    ids=((signs[:,0]>0).long()*4+(signs[:,1]>0).long()*2+(signs[:,2]>0).long())
    result={"events":len(y),"preferred_mass":float(data.joint_signal(y,c[:,None]).double().mean()),
        "condition_weighted_parity":float((data.condition_signal(c)*signs.prod(-1)).mean()),
        "near_corner_fraction":float(((y-signs).norm(dim=-1)<.5).double().mean()),
        "max_abs_coordinate":float(y.abs().max()),"conditions":{}}
    groups=[(str(condition),c==condition,condition) for condition in (-1,1)]
    if data.continuous:
        edges=torch.linspace(-1,1,data.condition_bins+1,dtype=c.dtype)
        groups=[(f"{float(lo):.6f}:{float(hi):.6f}" if data.condition_frequency is not None else f"{float(lo):.2f}:{float(hi):.2f}",
            (c>=lo)&((c<hi) if j<data.condition_bins-1 else (c<=hi)),None)
            for j,(lo,hi) in enumerate(zip(edges[:-1],edges[1:]))]
    for key,mask,condition in groups:
        yy=y[mask];ii=ids[mask]
        if not len(yy):raise ValueError("Both conditions required in diagnostics")
        counts=torch.bincount(ii,minlength=8)
        probs=counts.double()/len(yy)
        means=[];stds=[]
        for mode in range(8):
            local=yy[ii==mode]
            means.append(local.mean(0) if len(local) else torch.full((3,),float('nan')))
            stds.append(local.std(0,unbiased=False) if len(local) else torch.full((3,),float('nan')))
        means=torch.stack(means);stds=torch.stack(stds)
        all_modes=bool((counts>1).all())
        expected=data.probabilities(c[mask]).mean(0).double() if data.continuous else data.probabilities(condition).double()
        target=data.probabilities(c[mask],.9).mean(0).double() if data.continuous else data.probabilities(condition,.9).double()
        preferred=float(data.joint_signal(yy,c[mask,None]).mean())
        pc=data.positive_probability(c[mask]) if data.continuous else None
        expected_preferred=float(torch.where(data.condition_signal(c[mask])>=0,pc,1-pc).mean()) if data.continuous else data.mass
        result["conditions"][key]={"counts":counts.tolist(),"mode_probabilities":probs.tolist(),
            "condition_mean":float(c[mask].mean()),
            "preferred_mass":preferred,"expected_preferred_mass":expected_preferred,
            "preferred_mass_abs_error":abs(preferred-expected_preferred),
            "positive_parity_mass":float((yy.sign().prod(-1)>0).double().mean()),
            "expected_positive_parity_mass":float(expected[data.centers.prod(-1)>0].sum()),
            "target_positive_parity_mass":float(target[data.centers.prod(-1)>0].sum()),
            "reference_tv":float((probs-expected).abs().sum()/2),
            "target_tv":float((probs-target).abs().sum()/2),
            "centroid_error_max":float((means-data.centers).abs().max()) if all_modes else None,
            "width_min":float(stds.min()) if all_modes else None,
            "width_max":float(stds.max()) if all_modes else None,
            "marginal_positive":(yy>=0).double().mean(0).tolist()}
    return result


def baseline_gate(metrics):
    return metrics["near_corner_fraction"]>=.90 and all(
        v.get("preferred_mass_abs_error",abs(v["preferred_mass"]-.1))<=.04 and v["reference_tv"]<=.08 and
        v["centroid_error_max"] is not None and v["centroid_error_max"]<=.12 and
        v["width_min"]>=.075 and v["width_max"]<=.27
        for v in metrics["conditions"].values())


@torch.no_grad()
def generation_panel(model,data,cfg,seed,n=None):
    n=cfg.eval_events if n is None else n
    rng=generator(seed)
    # Exactly balanced conditions, independent latent candidates.
    c=((torch.arange(n,dtype=torch.float32)+.5)/n*2-1)[:,None] if data.continuous else torch.cat((-torch.ones(n//2,1),torch.ones(n-n//2,1)))
    z=torch.randn(n,cfg.candidates,3,generator=rng)
    ys=[]
    for cc,zz in zip(c.split(128),z.split(128)):
        ys.append(ddim(model,cc[:,None],zz,cfg.ddim_steps))
    y=torch.cat(ys);ctx=c[:,None].expand(-1,cfg.candidates,-1)
    reward=ModeReward(data.mass)(y,ctx,data)
    hits=data.joint_signal(y,ctx)
    metrics=cube_metrics(y.flatten(0,1),ctx.flatten(0,1),data)
    metrics.update(group_hit_fraction=float(hits.amax(1).mean()),reward_mean=float(reward.mean()))
    return reward,hits,metrics


def prepare_splits(data,cfg,seed):
    splits={}
    for i,(name,n) in enumerate((("train",cfg.train_events),("validation",cfg.validation_events),("test",cfg.test_events))):
        cs=[];ys=[]
        # Preserve the original32k training prefix, with new independent blocks.
        for block,start in enumerate(range(0,n,32768)):
            rng=generator(seed+610000+i+1000*block)
            c=data.contexts(min(32768,n-start),rng)
            cs.append(c);ys.append(data.sample(c,rng))
        splits[name]={"condition":torch.cat(cs),"target":torch.cat(ys)}
    return splits


def run(output,policy_steps=300,patience=20,min_delta=1e-4,seed=17,wandb_mode="offline",train_events=32768,continuous_condition=False,reference_sharpness=1.,pretrain_only=False,condition_frequency=None):
    if min(policy_steps,patience)<1 or min_delta<0:raise ValueError("Invalid budget")
    if train_events<32768 or train_events%32768:
        raise ValueError("Use a positive multiple of32768 to preserve the original training prefix")
    cfg=replace(Config(),dimensions=3,context_dim=1,hidden=128,train_events=train_events,
        validation_events=8192,test_events=8192,ddim_steps=50,eval_events=4096,
        policy_steps=policy_steps,eval_every=50)
    data=CubeDistribution(cfg,continuous=continuous_condition,reference_sharpness=reference_sharpness,condition_frequency=condition_frequency)
    output.mkdir(parents=True,exist_ok=False)
    report={"state":"pretraining","config":asdict(cfg),"seed":seed,"reference_mass":.1,
        "target_mass":.9,"width":.15,"patience":patience,"min_delta":min_delta,
        "continuous_condition":continuous_condition,
        "condition_frequency":condition_frequency,
        "reference_sharpness":reference_sharpness,
        "condition_distribution":"Uniform[-1,1]" if continuous_condition else "binary +/-1",
        "positive_parity_probability":"reference sigmoid(sharpness*logit(.5-.4*g)); target .5+.4*g; g=sin(k*pi*c) when frequency set, otherwise c" if continuous_condition else "binary preferred mass .1 / .9",
        "training_input":"raw noisy y, time, scalar condition; no parity/Fourier/transport",
        "reward":"nearest-corner categorical log target/reference ratio, conditional on c; not exact neural density ratio",
        "selection":"best validation velocity MSE; early stopping; no fixed epoch budget",
        "primary":"Independent paired preferred-mode-mass gain, with corner shape and all eight probabilities",
        "arms":{}}
    atomic_json(output/"report.json",report)
    import wandb
    wb=wandb.init(project="dgpo-toy",mode=wandb_mode,dir=str(output.resolve()),
        name=(f"Can diffusion learn switching conditions? | cube k={condition_frequency} | pretrain" if condition_frequency is not None else "Can diffusion move mode probability? | conditional cube | oracle reward | V MSE 0 vs 1"),
        group="Conditional cube transport",config=report,tags=["toy","no-classifier","raw-no-ema"])
    report["wandb"]={"id":wb.id,"mode":wandb_mode,"directory":wb.dir}
    splits=prepare_splits(data,cfg,seed)
    atomic_checkpoint(output/"dataset.pt",{"splits":splits,"config":asdict(cfg),"seed":seed,"mass":.1,"width":.15,"continuous_condition":continuous_condition,"condition_frequency":condition_frequency,"reference_sharpness":reference_sharpness})
    torch.manual_seed(seed);model=Denoiser(cfg)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.fit_lr,weight_decay=cfg.weight_decay)
    rng=generator(seed+620000);val=make_panel(splits["validation"],seed+630000)
    best=float('inf');anchor=float('inf');stale=0;epoch=0;step=0;best_weights=None
    try:
        with (output/"progress.jsonl").open("w") as log:
            def emit(row):
                line=json.dumps(row,allow_nan=False);log.write(line+"\n");log.flush();print(line,flush=True)
                prefix=row["phase"]+"/"
                wb.log(flatten_metrics(row,prefix))
            while stale<patience:
                epoch+=1;order=torch.randperm(cfg.train_events,generator=rng);total=0.
                model.train()
                for idx in order.split(cfg.fit_batch):
                    c=splits["train"]["condition"][idx];y=splits["train"]["target"][idx]
                    x,t,target=noisy_target(y,rng)
                    loss=(model(x,t,c)-target).square().mean()
                    opt.zero_grad(set_to_none=True);loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
                    step+=1;total+=float(loss.detach())*len(idx)
                model.eval();score=validation(model,val)["velocity_mse"]
                if score<best:
                    best=score;best_weights=copy.deepcopy(model.state_dict())
                    report.update(best_epoch=epoch,best_step=step,best_validation_mse=best)
                    atomic_checkpoint(output/"best_pretrain.pt",{"model":best_weights,"config":asdict(cfg),"epoch":epoch,"step":step,"continuous_condition":continuous_condition,"condition_frequency":condition_frequency,"reference_sharpness":reference_sharpness})
                anchor,stale=early_stop_update(score,anchor,stale,min_delta)
                row={"phase":"pretrain","epoch":epoch,"step":step,"train_mse":total/cfg.train_events,
                    "val_mse":score,"best_val_mse":best,"stale_epochs":stale}
                if epoch==1 or epoch%10==0:row["generation"]=generation_panel(model,data,cfg,640017,n=1024)[2]
                emit(row)
                report.update(epoch=epoch,pretrain_step=step,stale_epochs=stale)
                atomic_json(output/"report.json",report)
                if epoch%5==0:
                    atomic_checkpoint(output/"pretrain_state.pt",{"model":model.state_dict(),"optimizer":opt.state_dict(),
                        "rng":rng.get_state(),"epoch":epoch,"step":step,"best":best_weights,"stale":stale,"anchor":anchor})
            model.load_state_dict(best_weights);model.eval()
            report["test_velocity_mse"]=validation(model,make_panel(splits["test"],seed+630001))["velocity_mse"]
            _,_,baseline=generation_panel(model,data,cfg,650017)
            report["baseline"]=baseline;report["baseline_gate"]=baseline_gate(baseline)
            emit({"phase":"baseline_gate","passed":report["baseline_gate"],"generation":baseline})
            if pretrain_only:
                report["state"]="pretrain_completed";return report
            if not report["baseline_gate"]:
                report["state"]="baseline_gate_failed";return report
            atomic_checkpoint(output/"baseline.pt",{"model":model.state_dict(),"config":asdict(cfg),"baseline":baseline,"continuous_condition":continuous_condition})
            report["state"]="training_dgpo";atomic_json(output/"report.json",report)
            reference_state=copy.deepcopy(model.state_dict());reward=ModeReward(.1)
            base_r,base_h,base_stats=generation_panel(model,data,cfg,660017)
            report["endpoint_initial"]=base_stats
            results={}
            for name,coefficient in (("no_penalty",0.),("velocity_mse_1",1.)):
                def checkpoint(s,m,o,r,h):
                    if s==1 or s%cfg.eval_every==0 or s==policy_steps:
                        _,_,stats=generation_panel(m,data,cfg,640017)
                        emit({"phase":name+"_generation","step":s,**stats})
                        atomic_checkpoint(output/(name+".pt"),{"model":m.state_dict(),"optimizer":o.state_dict(),
                            "rng":r.get_state(),"step":s,"config":asdict(cfg),"velocity_coefficient":coefficient,"continuous_condition":continuous_condition})
                def policy_emit(row):
                    emit({**row,"phase":name+"_policy"})
                final,history=policy_train("dgpo",model,reward,data,cfg,seed,670017,policy_emit,
                    checkpoint_callback=checkpoint,velocity_coefficient=coefficient)
                r,h,stats=generation_panel(final,data,cfg,660017)
                results[name]=(r,h)
                report["arms"][name]={"completed_steps":len(history),"endpoint":stats,
                    "preferred_mass_gain":paired_gain(h,base_h),"reward_gain":paired_gain(r,base_r)}
                atomic_json(output/"report.json",report)
            report["no_penalty_minus_velocity1"]=paired_gain(results["no_penalty"][1],results["velocity_mse_1"][1])
            report["reference_unchanged"]=all(torch.equal(v,reference_state[k]) for k,v in model.state_dict().items())
            report["state"]="completed"
    except BaseException as exc:
        report.update(state="failed_or_interrupted",error=repr(exc));raise
    finally:
        atomic_json(output/"report.json",report);wb.summary.update(flatten_metrics(report));wb.summary["state"]=report["state"]
        wb.finish(exit_code=1 if report["state"]=="failed_or_interrupted" else 0)
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--policy-steps",type=int,default=300)
    p.add_argument("--patience",type=int,default=20)
    p.add_argument("--min-delta",type=float,default=1e-4)
    p.add_argument("--seed",type=int,default=17)
    p.add_argument("--train-events",type=int,default=32768)
    p.add_argument("--continuous-condition",action="store_true")
    p.add_argument("--reference-sharpness",type=float,default=1.)
    p.add_argument("--pretrain-only",action="store_true")
    p.add_argument("--condition-frequency",type=int)
    p.add_argument("--wandb-mode",choices=("offline","online","disabled"),default="offline")
    args=p.parse_args();torch.set_num_threads(1);run(**vars(args))
