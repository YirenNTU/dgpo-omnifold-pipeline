"""One new coefficient=.1 arm; reuse saved 0/1 controls and diagnose endpoint repair."""
import argparse
import json
from pathlib import Path
import torch
from .velocity_kl import run as train
from .fixed_reward import load_source
from .conditional import Denoiser, Distribution
from .marginal_repair import sample, fit_maps, transport, metrics
from .structure_metrics import paired_structure_change, structure_target
from .truth_pretrain import atomic_json


def run(root, output):
    source=root/"low_ess_joint3_v1/reward.pt"
    control=root/"fixed_reward_joint3_resume10k_v1"
    one=root/"velocity_kl1_joint3_10k_v1"
    if output.exists():
        raise ValueError("Use a separate output for this new coefficient arm")
    cfg,initial,_,_,_=load_source(source)
    saved=torch.load(one/"dgpo.pt",map_location="cpu",weights_only=True)
    evidence=json.loads((one/"report.json").read_text())
    if (saved["velocity_coefficient"]!=1. or saved["step"]!=10000
            or Path(saved["source"]).resolve()!=source.resolve()
            or evidence["seed"]!=17 or evidence["decision"] in ("running","interrupted_or_error")):
        raise ValueError("Requires completed matched coefficient1 control")
    # Existing runner validates no-KL source/config/seed and reproduces its endpoint.
    result=train(source,control,output,coefficient=.1,policy_steps=10000,seed=17,endpoint_seed=94017)
    if saved["config"]!=result["config"]:
        raise ValueError("Coefficient1 control config mismatch")
    models={"initial":initial}
    for name,path in (("lambda0",control),("lambda01",output),("lambda1",one)):
        state=torch.load(path/"dgpo.pt",map_location="cpu",weights_only=True)
        model=Denoiser(cfg);model.load_state_dict(state["model"])
        models[name]=model.eval().requires_grad_(False)
    data=Distribution(cfg)
    diagnostics={"state":"evaluating","calibration_seed":340017,"evaluation_seed":350017,
        "calibration_events":32768,"evaluation_events":16384,"cases":{},
        "scope":"Pooled oracle repair diagnostic only; no transport in policy training; fixed calibration map uncertainty."}
    panels={}
    for name,model in models.items():
        c,y=sample(model,data,32768,340017)
        maps=fit_maps((y-data.mean(c)).double().numpy())
        c,y=sample(model,data,16384,350017)
        repaired=torch.from_numpy(transport((y-data.mean(c)).double().numpy(),maps)).float()+data.mean(c)
        row={}
        for label,values in (("raw",y),("repaired",repaired)):
            panels[name,label],row[label]=metrics(c,values,data)
            if name!="initial":
                row[label+"_vs_initial"]=paired_structure_change(panels[name,label],panels["initial",label],structure_target(data),350018)
        m=row["raw"]
        row["low_order_gate"]=m["mean_absmax"]<.05 and m["variance_error_absmax"]<.1 and m["pair_covariance_absmax"]<.05
        if name!="initial":
            row["joint_and_low_order_pass"]=row["low_order_gate"] and row["raw_vs_initial"]["hi95"]<0
        diagnostics["cases"][name]=row
        atomic_json(output/"repair_diagnostic.json",diagnostics)
    diagnostics["state"]="completed"
    atomic_json(output/"repair_diagnostic.json",diagnostics)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("artifacts/dgpo_toy"))
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(1);run(**vars(args))
