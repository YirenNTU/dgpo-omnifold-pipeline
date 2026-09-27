"""Read-only fixed-classifier structural direction audit on fresh toy samples."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scipy.stats import rankdata
from .conditional import Config,Distribution,Denoiser,ddim,generator,paired_gain
from .low_ess_recovery import JointFourierClassifier
from .direct_transport import FixedTransport
from .broad_coverage import region_hits
from .structure_metrics import structure_features,structure_target,paired_structure_change
from .truth_pretrain import atomic_json


def nominal_ratio(y,c,data,amplitude):
    phase=data.angles(y,c).reshape(*y.shape[:-1],-1,3).sum(-1)-data.phase(c)
    return data.nominal_log_ratio(y,c)-torch.log1p(amplitude*phase.cos()).sum(-1)


def candidate_metrics(rewards,oracle,hits,features,data,seed):
    a=rewards.double();b=oracle.double()
    ac=a-a.mean(1,keepdim=True);bc=b-b.mean(1,keepdim=True)
    ranks_a=torch.from_numpy(rankdata(a.numpy(),axis=1))
    ranks_b=torch.from_numpy(rankdata(b.numpy(),axis=1))
    ra=ranks_a-ranks_a.mean(1,keepdim=True);rb=ranks_b-ranks_b.mean(1,keepdim=True)
    denom=ra.norm(dim=1)*rb.norm(dim=1);valid=denom>0
    rho=(ra[valid]*rb[valid]).sum(1)/denom[valid]
    index=a.argmax(1);rows=torch.arange(len(a))
    selected=hits[rows,index][:,None];random=hits.mean(1,keepdim=True)
    selected_oracle=b[rows,index][:,None];mean_oracle=b.mean(1,keepdim=True)
    # Within-context finite-K self-normalized weighting is a support diagnostic,
    # not a generator update or an unbiased population density-ratio estimate.
    raw_features=features.mean(1).double().numpy()
    weighted=(features.double()*a.softmax(1)[...,None]).sum(1).numpy()
    oracle_weighted=(features.double()*b.softmax(1)[...,None]).sum(1).numpy()
    target=structure_target(data)
    return {"contexts":len(a),"k":a.shape[1],
        "centered_logit_oracle_cosine":float((ac*bc).sum()/(ac.norm()*bc.norm())),
        "mean_within_context_spearman":float(rho.mean()),"rank_valid_fraction":float(valid.double().mean()),
        "random_hit_rate":float(random.mean()),"top_score_hit_rate":float(selected.mean()),
        "top_oracle_hit_rate":float(hits[rows,b.argmax(1)].mean()),
        "top_score_hit_gain":paired_gain(selected,random),
        "top_score_nominal_logratio_gain":paired_gain(selected_oracle,mean_oracle),
        "raw_moment_rmse":float(np.sqrt(((raw_features.mean(0)-target)**2).mean())),
        "classifier_reweighted_moment_rmse":float(np.sqrt(((weighted.mean(0)-target)**2).mean())),
        "oracle_reweighted_moment_rmse":float(np.sqrt(((oracle_weighted.mean(0)-target)**2).mean())),
        "classifier_reweight_change":paired_structure_change(weighted,raw_features,target,seed),
        "oracle_reweight_change":paired_structure_change(oracle_weighted,raw_features,target,seed)}


@torch.no_grad()
def run(directory,output,n=8192):
    saved=torch.load(directory/"reward.pt",map_location="cpu",weights_only=True)
    cfg=Config(**saved["config"]);data=Distribution(cfg)
    critic=JointFourierClassifier(cfg);critic.load_state_dict(saved["classifier"])
    critic.eval().requires_grad_(False)
    initial=Denoiser(cfg);initial.load_state_dict(saved["initial"]);initial.eval()
    transform=FixedTransport(data,saved["amplitude"])
    output.mkdir(parents=True,exist_ok=False)
    report={"state":"evaluating","source":str(directory.resolve()),"seed":470017,
        "classifier":"frozen validation-selected strong; no refit",
        "scope":"Structural preference, not exact actual-policy ratio calibration, gradient alignment, or DGPO convergence. Nominal ratio uses ideal Gaussian base.",
        "models":{}}
    models={"initial":(initial,0)}
    if (directory/"dgpo_state.pt").exists():
        state=torch.load(directory/"dgpo_state.pt",map_location="cpu",weights_only=True)
        if state["amplitude"]!=saved["amplitude"]:raise ValueError("Transport mismatch")
        current=Denoiser(cfg);current.load_state_dict(state["model"]);current.eval()
        models["latest"]=(current,state["step"])
    for name,(model,step) in models.items():
        rng=generator(470017);c=data.contexts(n,rng);noise=torch.randn(n,8,cfg.dimensions,generator=rng)
        scores=[];ratios=[];hits=[];features=[]
        for cc,z in zip(c.split(128),noise.split(128)):
            ctx=cc[:,None].expand(-1,8,-1);y=transform(ddim(model,cc[:,None],z,cfg.ddim_steps),ctx)
            scores.append(critic(y,ctx,data));ratios.append(nominal_ratio(y,ctx,data,saved["amplitude"]))
            hits.append(region_hits(y,ctx,data).float());features.append(structure_features(y,ctx,data))
        report["models"][name]={"policy_step":step,**candidate_metrics(*[torch.cat(v) for v in (scores,ratios,hits,features)],data,470018)}
    # Same contexts/noise, ideal input marginals: isolate known joint alterations.
    rng=generator(480017);c=data.contexts(n*8,rng);z=torch.randn(n*8,cfg.dimensions,generator=rng)
    case_scores={};oracle_scores={};report["structural_controls"]={}
    for name,amp,wrong in (("gaussian",0.,False),("correct_a06",.6,False),("correct_a09",.9,False),("wrong_phase_a06",.6,True)):
        chunks=[];ys=[]
        for cc,zz in zip(c.split(512),z.split(512)):
            y=FixedTransport(data,amp)(data.mean(cc)+zz,cc)
            if wrong:
                angles=data.angles(y,cc).reshape(len(y),-1,3)
                angles[...,2]=(angles[...,2]+torch.pi).remainder(2*torch.pi)
                u=(angles.flatten(1)/(2*torch.pi)).clamp(1e-7,1-1e-7)
                y=data.mean(cc)+2**.5*torch.erfinv(2*u-1)
            chunks.append(critic(y,cc,data));ys.append(y)
        score=torch.cat(chunks);y=torch.cat(ys);res=y-data.mean(c);centered=res-res.mean(0);cov=centered.T@centered/len(res)
        case_scores[name]=score[:,None]
        oracle=nominal_ratio(y,c,data,saved["amplitude"])
        oracle_scores[name]=oracle[:,None]
        report["structural_controls"][name]={"reward_mean":float(score.mean()),
            "nominal_oracle_reward_mean":float(oracle.mean()),
            "joint_mean":float(data.joint_signal(y,c).mean()),"mean_absmax":float(res.mean(0).abs().max()),
            "variance_error":float((cov.diag()-1).abs().max()),"pair_covariance":float((cov-torch.diag(cov.diag())).abs().max())}
    report["control_contrasts"]={name:paired_gain(case_scores[a],case_scores[b]) for name,a,b in
        (("more_correct_joint","correct_a09","correct_a06"),("correct_vs_wrong_phase","correct_a06","wrong_phase_a06"),
         ("correct_vs_gaussian","correct_a06","gaussian"))}
    report["nominal_oracle_control_contrasts"]={name:paired_gain(oracle_scores[a],oracle_scores[b]) for name,a,b in
        (("more_correct_joint","correct_a09","correct_a06"),("correct_vs_wrong_phase","correct_a06","wrong_phase_a06"),
         ("correct_vs_gaussian","correct_a06","gaussian"))}
    report["state"]="completed";atomic_json(output/"report.json",report)
    print(json.dumps(report,allow_nan=False),flush=True)
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory",type=Path,default=Path("artifacts/dgpo_toy/direct_transport_joint_v1"))
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(1);run(**vars(args))
