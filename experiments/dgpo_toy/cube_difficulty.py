"""One difficulty increase, verified actual coverage before oracle DGPO."""
import argparse
import json
from dataclasses import replace
from pathlib import Path
import torch
from .conditional import Config,Denoiser,weight_health
from .parity_cube import CubeDistribution,generation_panel,run as pretrain
from .cube_oracle_policy import run as policy
from .truth_pretrain import atomic_json


def measure(source,seed=720017):
    meta=json.loads((source/"report.json").read_text())
    saved=torch.load(source/"best_pretrain.pt",map_location="cpu",weights_only=True)
    cfg=replace(Config(**saved["config"]),eval_events=4096)
    data=CubeDistribution(cfg,continuous=True,reference_sharpness=meta.get("reference_sharpness",1.),condition_frequency=meta.get("condition_frequency"))
    model=Denoiser(cfg);model.load_state_dict(saved["model"]);model.eval()
    r,_,stats=generation_panel(model,data,cfg,seed)
    c=(torch.arange(10000,dtype=torch.float64)+.5)/10000*2-1
    q=data.probabilities(c);p=data.probabilities(c,.9)
    theoretical_ess=float(1/(p.square()/q).sum(-1).mean())
    stats.update(oracle_weight_health=weight_health(r),ideal_data_ratio_ess_fraction=theoretical_ess,
        ideal_preferred_mass=float(torch.where(c>=0,data.positive_probability(c),1-data.positive_probability(c)).mean()))
    return stats


def run(output,easy_source,sharpness=4.):
    output.mkdir(parents=True,exist_ok=False)
    report={"state":"pretraining_harder_reference","sharpness":sharpness,
        "easy_source":str(easy_source.resolve()),"scope":"Change reference probabilities only; truth fixed. New pretraining weights and ratio reward differ, so not ESS-only causality.",
        "difficulty_rule":"actual preferred mass <= half easy, actual oracle-weight ESS <= .75 easy, near-corner >=.9; no reference-fit gate"}
    atomic_json(output/"report.json",report)
    try:
        pretrain(output/"pretrain",train_events=131072,continuous_condition=True,
            reference_sharpness=sharpness,pretrain_only=True)
        easy=measure(easy_source);hard=measure(output/"pretrain")
        passed=(hard["preferred_mass"]<=.5*easy["preferred_mass"] and
            hard["oracle_weight_health"]["ess_fraction"]<=.75*easy["oracle_weight_health"]["ess_fraction"] and
            hard["near_corner_fraction"]>=.9)
        report.update(easy=easy,hard=hard,difficulty_established=passed)
        if not passed:
            report["state"]="difficulty_not_established";return report
        report["state"]="training_oracle_dgpo";atomic_json(output/"report.json",report)
        result=policy(output/"pretrain",output/"policy",steps=300)
        report.update(state="completed",policy_report=str((output/"policy/report.json").resolve()),
            policy_arms=result["arms"])
    except BaseException as exc:
        report.update(state="failed_or_interrupted",error=repr(exc));raise
    finally:atomic_json(output/"report.json",report)
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--easy-source",type=Path,default=Path("artifacts/dgpo_toy/parity_cube_continuous_4x_v1"))
    p.add_argument("--sharpness",type=float,default=4.)
    args=p.parse_args();torch.set_num_threads(1);run(**vars(args))
