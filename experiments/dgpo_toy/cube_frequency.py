"""Single-knob conditional-frequency sweep, oracle DGPO plus velocity MSE one."""
import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import torch
from .conditional import Config, Denoiser, policy_train, paired_gain, weight_health
from .parity_cube import CubeDistribution, ModeReward, generation_panel, run as pretrain
from .coverage_budget import flatten_metrics
from .truth_pretrain import atomic_json, atomic_checkpoint


def ideal_metrics(k, n=65536):
    c=(torch.arange(n,dtype=torch.float64)+.5)/n*2-1
    data=CubeDistribution(Config(dimensions=3,context_dim=1),continuous=True,
        reference_sharpness=4.,condition_frequency=k)
    p=data.positive_probability(c,truth=True);q=data.positive_probability(c)
    preferred=torch.where(data.condition_signal(c)>=0,q,1-q)
    return {"ess_fraction":float(1/(p.square()/q+(1-p).square()/(1-q)).mean()),
        "preferred_mass":float(preferred.mean()),
        "k8_group_hit":float((1-(1-preferred).pow(8)).mean()),
        "truth_preferred_mass":float((.5+.4*data.condition_signal(c).abs()).mean())}


def summarize(stats, rewards):
    bins=list(stats["conditions"].values())
    return {"preferred_mass":stats["preferred_mass"], "group_hit_fraction":stats["group_hit_fraction"],
        "near_corner_fraction":stats["near_corner_fraction"],
        "reward_mean":stats["reward_mean"], "reward_std":float(rewards.std(unbiased=False)), "weight":weight_health(rewards),
        "mean_target_tv":sum(b["target_tv"] for b in bins)/len(bins),
        "mean_reference_tv":sum(b["reference_tv"] for b in bins)/len(bins),
        "condition_weighted_parity":stats["condition_weighted_parity"],
        "truth_condition_weighted_parity":.4, "condition_bins":len(bins)}


def run_arm(source, output, k, wandb_mode):
    saved=torch.load(source/"best_pretrain.pt",map_location="cpu",weights_only=True)
    if saved.get("condition_frequency")!=k:raise ValueError("Frequency provenance mismatch")
    cfg=replace(Config(**saved["config"]),policy_steps=300,eval_every=50,eval_events=8192)
    data=CubeDistribution(cfg,continuous=True,reference_sharpness=4.,condition_frequency=k)
    model=Denoiser(cfg);model.load_state_dict(saved["model"]);model.eval()
    reward=ModeReward();output.mkdir(parents=True,exist_ok=False)
    report={"state":"initializing","frequency":k,"config":asdict(cfg),"source":str(source.resolve()),
        "source_step":saved["step"],"velocity_coefficient":1.,"ideal":ideal_metrics(k),
        "seeds":{"policy":17,"endpoint":760017,"monitor":770017,"native_monitor":780017},
        "scope":"Ideal categorical mixture oracle, not exact learned-generator density ratio. Actual coverage not assumed matched."}
    import wandb
    wb=wandb.init(project="dgpo-toy",mode=wandb_mode,dir=str(output.resolve()),
        name=f"Does condition complexity block reward? | cube k={k} | oracle | V MSE 1",
        group="Conditional cube frequency",config=report,tags=["oracle","frequency-sweep","no-classifier"])
    report["wandb"]={"id":wb.id,"mode":wandb_mode,"directory":wb.dir}
    try:
        br,bh,bs=generation_panel(model,data,cfg,760017)
        report["initial"]=summarize(bs,br);report["state"]="training_dgpo"
        atomic_json(output/"initial_conditions.json",bs)
        atomic_json(output/"report.json",report)
        with (output/"progress.jsonl").open("w") as log:
            def emit(row):
                line=json.dumps(row,allow_nan=False);print(line,flush=True);log.write(line+"\n");log.flush()
                wb.log(flatten_metrics(row,row["phase"]+"/"))
            def save(step,m,opt,rng,history):
                report["active_step"]=step
                if step==1 or step%50==0:
                    rr,_,ss=generation_panel(m,data,cfg,770017)
                    emit({"phase":"physical","step":step,**summarize(ss,rr)})
                    atomic_json(output/f"conditions_step{step}.json",ss)
                    atomic_checkpoint(output/"policy.pt",{"model":m.state_dict(),"optimizer":opt.state_dict(),
                        "rng":rng.get_state(),"history":history,"step":step,"config":asdict(cfg),
                        "condition_frequency":k,"reference_sharpness":4.,"velocity_coefficient":1.,
                        "source":str(source.resolve()),"seed":17,"monitor_seed":780017})
                atomic_json(output/"report.json",report)
            final,history=policy_train("dgpo",model,reward,data,cfg,17,780017,emit,save,velocity_coefficient=1.)
        report["completed_steps"]=len(history)
        fr,fh,fs=generation_panel(final,data,cfg,760017)
        report["endpoint"]=summarize(fs,fr)
        report["reward_gain"]=paired_gain(fr,br);report["preferred_mass_gain"]=paired_gain(fh,bh)
        bins=data.condition_bins
        report["reward_gain_by_condition"]={str(i):paired_gain(a,b) for i,(a,b) in enumerate(zip(fr.chunk(bins),br.chunk(bins)))}
        report["absorption"]="positive" if report["reward_gain"]["lo95"]>0 else "unresolved_or_negative"
        report["source_unchanged"]=all(torch.equal(v,saved["model"][key]) for key,v in model.state_dict().items())
        atomic_json(output/"endpoint_conditions.json",fs)
        atomic_checkpoint(output/"evaluation.pt",{"initial_reward":br,"final_reward":fr,"initial_hits":bh,"final_hits":fh})
        report["state"]="completed"
    except BaseException as exc:
        report.update(state="failed_or_interrupted",error=repr(exc));raise
    finally:
        atomic_json(output/"report.json",report);wb.summary.update(flatten_metrics(report))
        wb.finish(exit_code=0 if report["state"]=="completed" else 1)
    return report


def run(output,frequencies=(1,2,4,8),wandb_mode="offline"):
    if not frequencies or len(set(frequencies))!=len(frequencies) or any(k not in (1,2,4,8) for k in frequencies):
        raise ValueError("Choose distinct frequencies from 1,2,4,8")
    output.mkdir(parents=True,exist_ok=False)
    report={"state":"running","frequencies":list(frequencies),"arms":{},
        "question":"Does increasing only condition frequency suppress reward absorption?",
        "interpretation":"No automatic failure declaration from a CI spanning zero. Compare baseline fit and actual coverage before attributing reduced transfer to DGPO."}
    try:
        for k in frequencies:
            report.update(active_frequency=k,phase="pretraining");atomic_json(output/"report.json",report)
            source=output/f"k{k}"/"pretrain"
            fit=pretrain(source,train_events=131072,continuous_condition=True,reference_sharpness=4.,
                condition_frequency=k,pretrain_only=True,wandb_mode=wandb_mode)
            report["phase"]="policy";atomic_json(output/"report.json",report)
            arm=run_arm(source,output/f"k{k}"/"policy",k,wandb_mode)
            report["arms"][str(k)]={"pretrain_steps":fit["pretrain_step"],"selected_step":fit["best_step"],
                "test_velocity_mse":fit["test_velocity_mse"],**arm}
            atomic_json(output/"report.json",report)
        report["state"]="completed"
    except BaseException as exc:
        report.update(state="failed_or_interrupted",error=repr(exc));raise
    finally:atomic_json(output/"report.json",report)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--frequencies",type=int,nargs="+",default=[1,2,4,8])
    parser.add_argument("--wandb-mode",choices=("offline","online","disabled"),default="offline")
    args=parser.parse_args();torch.set_num_threads(1);run(**vars(args))
