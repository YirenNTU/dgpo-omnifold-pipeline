"""Independent-panel oracle marginal repair; no model fitting or policy updates."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scipy.special import ndtri, ndtr
from .conditional import Config, Denoiser, Distribution, ddim, generator
from .fixed_reward import load_source
from .structure_metrics import structure_features, structure_target, paired_structure_change
from .truth_pretrain import atomic_json


def fit_maps(z):
    maps = []
    for column in z.T:
        x, counts = np.unique(column, return_counts=True)
        p = (np.cumsum(counts)-counts/2)/len(column)
        maps.append((x, p))
    return maps


def transport(z, maps):
    return np.column_stack([ndtri(np.interp(column, x, p))
                           for column, (x,p) in zip(z.T, maps)])


@torch.no_grad()
def sample(model, data, n, seed):
    rng = generator(seed)
    c = data.contexts(n, rng)
    noise = torch.randn(n, data.cfg.dimensions, generator=rng)
    y = torch.cat([ddim(model, cc, zz, data.cfg.ddim_steps)
                   for cc,zz in zip(c.split(512),noise.split(512))])
    return c, y


def metrics(c, y, data):
    z = (y-data.mean(c)).double().numpy()
    n = len(z)
    covariance = np.cov(z, rowvar=False, bias=True)
    u = ndtr(np.sort(z, axis=0))
    ks = max(np.max(np.arange(1,n+1)[:,None]/n-u),
             np.max(u-np.arange(n)[:,None]/n))
    bins=[]
    for j in range(c.shape[1]):
        for low,high in [(-1,-.5),(-.5,0),(0,.5),(.5,1)]:
            subset=z[((c[:,j]>=low)&(c[:,j]<high)).numpy()]
            bins.append(float(np.max(np.abs(subset.mean(0)))))
    features=structure_features(y,c,data).double().numpy()
    summary={"moment_rmse":float(np.sqrt(np.mean((features.mean(0)-structure_target(data))**2))),
        "marginal_ks_max":float(ks), "mean_absmax":float(np.abs(z.mean(0)).max()),
        "variance_error_absmax":float(np.abs(np.diag(covariance)-1).max()),
        "pair_covariance_absmax":float(np.abs(covariance-np.diag(np.diag(covariance))).max()),
        "context_bin_mean_absmax":max(bins)}
    return features,summary


def run(output, root, ncal=32768, neval=16384):
    output.mkdir(parents=True,exist_ok=False)
    cfg, initial, _, _, _ = load_source(root/"low_ess_joint3_v1/reward.pt")
    cases={"old_initial":(cfg,initial)}
    for name, folder in [("old_pure_3k","fixed_reward_joint3_long_v1"),
                          ("old_pure_10k","fixed_reward_joint3_resume10k_v1")]:
        state=torch.load(root/folder/"dgpo_state.pt",map_location="cpu",weights_only=True)
        model=Denoiser(cfg);model.load_state_dict(state["model"])
        cases[name]=(cfg,model.eval())
    state=torch.load(root/"truth_diffusion_earlystop_v1/best_model.pt",map_location="cpu",weights_only=True)
    cfg=Config(**state["config"])
    initial=Denoiser(cfg);initial.load_state_dict(state["model"])
    cases["new_initial"]=(cfg,initial.eval())
    state=torch.load(root/"truth_three_arm_v1/strong_no_kl/state.pt",map_location="cpu",weights_only=True)
    model=Denoiser(cfg);model.load_state_dict(state["model"])
    cases["new_pure_10k"]=(cfg,model.eval())
    report={"state":"running","calibration_events":ncal,"evaluation_events":neval,
        "calibration_seed":240017,"evaluation_seed":250017,"cases":{},
        "scope":"Pooled residual marginal oracle; not conditional transport, covariance repair, or retrained diffusion.",
        "rule":"Support if repaired KS<.03, mean<.05, variance error<.10 AND repaired moment MSE vs repaired baseline upper95CI<0. Covariance and conditional bins diagnostic only.",
        "bootstrap_scope":"Evaluation uncertainty conditional on one fitted calibration map; no training-seed replication."}
    panels={}
    for name,(cfg,model) in cases.items():
        data=Distribution(cfg)
        cc,yy=sample(model,data,ncal,240017)
        maps=fit_maps((yy-data.mean(cc)).double().numpy())
        c,y=sample(model,data,neval,250017)
        z=(y-data.mean(c)).double().numpy()
        repaired=torch.from_numpy(transport(z,maps)).float()+data.mean(c)
        row={}
        for kind,values in [("raw",y),("repaired",repaired)]:
            panels[name,kind],row[kind]=metrics(c,values,data)
        row["calibration_unique_fraction_min"]=min(len(x)/ncal for x,p in maps)
        row["evaluation_outside_calibration_fraction"]=float(np.mean(np.column_stack(
            [(z[:,j]<x[0])|(z[:,j]>x[-1]) for j,(x,p) in enumerate(maps)])))
        row["repair_vs_raw"]=paired_structure_change(panels[name,"repaired"],panels[name,"raw"],structure_target(data),260017)
        if "pure" in name:
            baseline=name.split("_")[0]+"_initial"
            row["repaired_vs_repaired_initial"]=paired_structure_change(panels[name,"repaired"],panels[baseline,"repaired"],structure_target(data),260018)
            row["raw_vs_raw_initial"]=paired_structure_change(panels[name,"raw"],panels[baseline,"raw"],structure_target(data),260019)
            m=row["repaired"]
            row["marginal_gate"]=m["marginal_ks_max"]<.03 and m["mean_absmax"]<.05 and m["variance_error_absmax"]<.1
            row["supports_two_stage"]=row["marginal_gate"] and row["repaired_vs_repaired_initial"]["hi95"]<0
        report["cases"][name]=row
        print(json.dumps({"case":name,**row}),flush=True)
        atomic_json(output/"report.json",report)
    report["state"]="completed"
    atomic_json(output/"report.json",report)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path("artifacts/dgpo_toy"))
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    run(**vars(args))
