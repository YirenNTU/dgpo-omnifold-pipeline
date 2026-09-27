"""Second-stage learned weak reward from raw pure-strong10k, no marginal transport."""
import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
import torch
from .matched_h4 import run as fit, load_inputs
from .conditional import Classifier, Distribution, policy_train, policy_panel, paired_gain
from .structure_metrics import structure_panel, structure_target, paired_structure_change
from .truth_pretrain import atomic_json, atomic_checkpoint
from .fixed_reward import unchanged


def run(output, root, steps=10000):
    if steps < 1:
        raise ValueError("Positive second-stage update budget required")
    source=root/"truth_three_arm_v1/strong_no_kl/state.pt"
    dataset=root/"truth_dataset_seed17_v1/dataset.pt"
    cfg,_,initial,state=load_inputs(dataset,source,policy_source=True)
    cfg=replace(cfg,policy_steps=steps)
    output.mkdir(parents=True,exist_ok=False)
    report={"state":"fitting_classifier","source":str(source.resolve()),"source_step":10000,
        "config":asdict(cfg),"velocity_coefficient":1.,"reference":"second_stage_initial",
        "optimizer":"fresh AdamW; not full-state continuation","weights":"raw_no_ema",
        "classifier_seed":117,"policy_seed":117,"monitor_seed":330017,"endpoint_seed":332017,
        "criterion":"Endpoint marginal mean, variance error and pair covariance all lower than stage1; moment RMSE remains below original pretrained baseline. Paired moment CIs reported separately.",
        "caveat":"Weak sees Gaussian-CDF coordinates which can saturate at this extreme source; classifier AUC alone does not prove actionable within-context reward."}
    atomic_json(output/"report.json",report)
    try:
        result=fit(output/"classifier",dataset_path=dataset,diffusion_path=source,
            seed=117,feature_mode="plain",policy_source=True)
        saved=torch.load(output/"classifier/best_classifier.pt",map_location="cpu",weights_only=True)
        critic=Classifier(cfg,False);critic.load_state_dict(saved["classifier"])
        critic.eval().requires_grad_(False)
        data=Distribution(cfg)
        before_reward,_=policy_panel(initial,critic,data,cfg,332017)
        before_features,before=structure_panel(initial,data,cfg,332017)
        _,_,original,_=load_inputs(dataset,root/"truth_diffusion_earlystop_v1/best_model.pt")
        original_features,original_summary=structure_panel(original,data,cfg,332017)
        report.update(state="training_policy",classifier=result["selected_metrics"],
            classifier_best_step=saved["step"],stage1=before,original_baseline=original_summary)
        atomic_json(output/"report.json",report)
        with (output/"progress.jsonl").open("w") as log:
            def emit(row):
                log.write(json.dumps(row,allow_nan=False)+"\n");log.flush()
                if row["step"]==1 or row["step"]%1000==0 or row["step"]==steps:
                    print(json.dumps(row,allow_nan=False),flush=True)
            def checkpoint(step,model,opt,rng,history):
                if step==1 or step%250==0 or step==steps:
                    _,summary=structure_panel(model,data,cfg,330017)
                    emit({"phase":"structure","step":step,**summary})
                if step==1 or step%1000==0 or step==steps:
                    atomic_checkpoint(output/"state.pt",{"model":model.state_dict(),
                        "optimizer":opt.state_dict(),"rng":rng.get_state(),"history":history,
                        "step":step,"config":asdict(cfg),"seed":117,"monitor_seed":330017,
                        "source":str(source.resolve()),"velocity_coefficient":1.,"endpoint_controller":None})
            model,history=policy_train("dgpo",initial,critic,data,cfg,117,330017,emit,
                checkpoint,velocity_coefficient=1.)
        reward,_=policy_panel(model,critic,data,cfg,332017)
        features,summary=structure_panel(model,data,cfg,332017)
        report.update(state="completed",completed_steps=len(history),endpoint=summary,
            own_reward_gain=paired_gain(reward,before_reward),
            structure_vs_stage1=paired_structure_change(features,before_features,structure_target(data),332018),
            structure_vs_original=paired_structure_change(features,original_features,structure_target(data),332019),
            sources_unchanged=unchanged(initial,state["model"]) and unchanged(critic,saved["classifier"]))
        report["low_order_all_improve"]=all(summary[k]<before[k] for k in
            ("marginal_mean_absmax","marginal_variance_error_absmax","pair_covariance_absmax"))
        report["retains_high_order_point_improvement"]=summary["fourier_moment_rmse"]<original_summary["fourier_moment_rmse"]
    except BaseException as exc:
        report.update(state="failed_or_interrupted",error=repr(exc));raise
    finally:
        atomic_json(output/"report.json",report)
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("artifacts/dgpo_toy"))
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--steps",type=int,default=10000)
    args=p.parse_args();torch.set_num_threads(1);run(**vars(args))
