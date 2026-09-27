"""Explicit imperfect-baseline oracle-reward DGPO diagnostic; no classifier."""
import argparse
import json
from dataclasses import asdict,replace
from pathlib import Path

import torch
from .conditional import Config,Denoiser,policy_train,paired_gain
from .parity_cube import CubeDistribution,ModeReward,generation_panel,baseline_gate
from .coverage_budget import flatten_metrics
from .truth_pretrain import atomic_json,atomic_checkpoint


def paired_bins(after,before):
    c=(torch.arange(len(after),dtype=torch.float32)+.5)/len(after)*2-1
    return {f"{lo:.2f}:{lo+.25:.2f}":paired_gain(after[(c>=lo)&(c<lo+.25)],before[(c>=lo)&(c<lo+.25)])
        for lo in (-1.,-.75,-.5,-.25,0.,.25,.5,.75)}


def run(source,output,steps=300,wandb_mode="offline"):
    if steps<1:raise ValueError("Positive policy budget required")
    saved=torch.load(source/"best_pretrain.pt",map_location="cpu",weights_only=True)
    if not saved.get("continuous_condition",False):raise ValueError("Expected continuous cube checkpoint")
    source_report=json.loads((source/"report.json").read_text())
    if source_report.get("condition_frequency") is not None:
        raise ValueError("Use cube_frequency for frequency-aware policy evaluation")
    cfg=replace(Config(**saved["config"]),policy_steps=steps,eval_every=50)
    data=CubeDistribution(cfg,source_report["reference_mass"],source_report["width"],continuous=True,reference_sharpness=source_report.get("reference_sharpness",1.))
    initial=Denoiser(cfg);initial.load_state_dict(saved["model"]);initial.eval()
    reward=ModeReward(data.mass)
    output.mkdir(parents=True,exist_ok=False)
    report={"state":"initializing","source":str((source/"best_pretrain.pt").resolve()),
        "source_epoch":saved["epoch"],"source_pretrain_step":saved["step"],"config":asdict(cfg),
        "classifier":None,"new_pretrain_steps":0,"reference":"fixed source checkpoint, fresh AdamW per arm",
        "acceptance":"User authorized imperfect baseline; original reference-fit gate retained as diagnostic only",
        "reference_sharpness":data.reference_sharpness,
        "reward":"log target-mode probability / configured data-reference-mode probability; fixed throughout RL",
        "scope":"Ideal categorical data-reference ratio, NOT exact ratio to actual learned neural diffusion",
        "primary":"Paired independent reward/preferred-mode-mass gains by condition; all eight mode probabilities and corner shape reported separately",
        "seeds":{"policy":17,"native_monitor":710017,"physical_monitor":700017,"endpoint":690017},
        "arms":{}}
    atomic_json(output/"report.json",report)
    import wandb
    wb=wandb.init(project="dgpo-toy",mode=wandb_mode,dir=str(output.resolve()),
        name="Can imperfect diffusion absorb reward? | continuous cube | oracle ratio | V MSE 0 vs 1",
        group="Conditional cube transport",config=report,tags=["oracle-reward","no-classifier","imperfect-baseline"])
    report["wandb"]={"id":wb.id,"directory":wb.dir,"mode":wandb_mode}
    try:
        base_r,base_h,base_stats=generation_panel(initial,data,cfg,690017)
        report["initial"]=base_stats;report["reference_fit_gate_diagnostic"]=baseline_gate(base_stats)
        if not torch.isfinite(base_r).all():raise FloatingPointError("Nonfinite baseline")
        # This is NOT a reference-fidelity gate: missing modes only prevent interpreting shape metrics.
        report["all_modes_observed"]=all(min(v["counts"])>1 for v in base_stats["conditions"].values())
        report["state"]="training_dgpo";atomic_json(output/"report.json",report)
        endpoints={}
        with (output/"progress.jsonl").open("w") as log:
            def emit(row):
                line=json.dumps(row,allow_nan=False);log.write(line+"\n");log.flush();print(line,flush=True)
                wb.log(flatten_metrics(row,row["phase"]+"/"))
            for name,coefficient in (("no_penalty",0.),("velocity_mse_1",1.)):
                report["active_arm"]=name
                def save(step,model,opt,rng,history):
                    report["active_step"]=step
                    if step==1 or step%cfg.eval_every==0 or step==steps:
                        _,_,stats=generation_panel(model,data,cfg,700017)
                        emit({"phase":name+"_physical","step":step,**stats})
                        atomic_checkpoint(output/(name+".pt"),{"model":model.state_dict(),"optimizer":opt.state_dict(),
                            "rng":rng.get_state(),"history":history,"step":step,"config":asdict(cfg),
                            "velocity_coefficient":coefficient,"continuous_condition":True,"source":report["source"],
                            "seed":17,"monitor_seed":710017})
                    atomic_json(output/"report.json",report)
                model,history=policy_train("dgpo",initial,reward,data,cfg,17,710017,
                    lambda row:emit({**row,"phase":name+"_policy"}),save,velocity_coefficient=coefficient)
                r,h,stats=generation_panel(model,data,cfg,690017)
                endpoints[name]=(r,h)
                report["arms"][name]={"steps":len(history),"endpoint":stats,
                    "reward_gain":paired_gain(r,base_r),"preferred_mass_gain":paired_gain(h,base_h),
                    "reward_gain_by_condition":paired_bins(r,base_r),
                    "preferred_mass_gain_by_condition":paired_bins(h,base_h)}
                atomic_json(output/"report.json",report)
        report["no_penalty_minus_velocity1"]={"reward":paired_gain(endpoints["no_penalty"][0],endpoints["velocity_mse_1"][0]),
            "preferred_mass":paired_gain(endpoints["no_penalty"][1],endpoints["velocity_mse_1"][1])}
        report["source_weights_unchanged"]=all(torch.equal(v,saved["model"][k]) for k,v in initial.state_dict().items())
        report["state"]="completed"
    except BaseException as exc:
        report.update(state="failed_or_interrupted",error=repr(exc));raise
    finally:
        atomic_json(output/"report.json",report);wb.summary.update(flatten_metrics(report));wb.summary["state"]=report["state"]
        wb.finish(exit_code=0 if report["state"]=="completed" else 1)
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,default=Path("artifacts/dgpo_toy/parity_cube_continuous_4x_v1"))
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--steps",type=int,default=300)
    p.add_argument("--wandb-mode",choices=("offline","online","disabled"),default="offline")
    args=p.parse_args();torch.set_num_threads(1);run(**vars(args))
