"""Broader correctly centered joint targets; ordinary diffusion pretraining only."""
import argparse
from dataclasses import replace, asdict
from pathlib import Path
import torch
from .conditional import Config, Distribution, Denoiser, generator, ddim, paired_gain
from .truth_pretrain import run as pretrain, atomic_json
from .structure_metrics import structure_panel, structure_target, paired_structure_change


def region_hits(y,c,data,width=1.):
    phase=data.angles(y,c).reshape(*y.shape[:-1],-1,3).sum(-1)-data.phase(c)
    wrapped=torch.atan2(phase.sin(),phase.cos())
    return (wrapped.abs()<width).all(-1)


@torch.no_grad()
def coverage(model,data,cfg,seed=370017,oracle=False):
    rng=generator(seed)
    c=data.contexts(cfg.eval_events,rng)
    noise=torch.randn(cfg.eval_events,128,cfg.dimensions,generator=rng)
    hits=[]
    for cc,z in zip(c.split(64),noise.split(64)):
        ctx=cc[:,None].expand(-1,128,-1)
        y=(data.sample(ctx.reshape(-1,cfg.context_dim),rng,truth=True).reshape_as(z)
           if oracle else ddim(model,cc[:,None],z,cfg.ddim_steps))
        hits.append(region_hits(y,ctx,data))
    hits=torch.cat(hits).float()
    return {"per_candidate":hits.mean(1),"any_k8":hits[:,:8].amax(1),"any_k128":hits.amax(1)}


def run(output,root):
    output.mkdir(parents=True,exist_ok=False)
    cfg=replace(Config(),kappa=2.)
    report={"state":"pretraining","source_target_kappa":8.,"broad_target_kappa":2.,
        "structured_mass":.9,"config":asdict(cfg),"region":"all four wrapped triple-phase errors abs<1 rad",
        "primary":"paired any-K8 coverage lower95CI>0 vs existing truth-trained baseline; mean<.05,varerror<.15,paircov<.05",
        "caveat":"Known toy phase prior changes training distribution, not an ESS-only intervention. Broader target need not yield better learned coverage. No classifier or RL until gate passes."}
    atomic_json(output/"report.json",report)
    try:
        fit=pretrain(output/"diffusion",dataset_path=output/"dataset.pt",cfg=cfg,seed=17,
            patience=20,min_delta=1e-4)
        models={}
        for name,path in (("baseline",root/"truth_diffusion_earlystop_v1/best_model.pt"),
                          ("broad",output/"diffusion/best_model.pt")):
            state=torch.load(path,map_location="cpu",weights_only=True)
            expected={**state["config"],"kappa":2.}
            if expected!=asdict(cfg) or state["weights"]!="raw":
                raise ValueError("Baseline must match architecture and setup apart from kappa")
            model=Denoiser(cfg);model.load_state_dict(state["model"])
            models[name]=model.eval().requires_grad_(False)
        truth=Distribution(replace(cfg,kappa=8.))
        panels={name:coverage(model,truth,cfg) for name,model in models.items()}
        # Analytic samplers validate only the proposed targets, not learned performance.
        panels["oracle_truth"]=coverage(None,truth,cfg,seed=371017,oracle=True)
        panels["oracle_broad"]=coverage(None,Distribution(cfg),cfg,seed=372017,oracle=True)
        report["coverage"]={n:{k:float(v.mean()) for k,v in values.items()} for n,values in panels.items()}
        report["paired_coverage_change"]={k:paired_gain(panels["broad"][k][:,None],panels["baseline"][k][:,None]) for k in panels["broad"]}
        structures={}
        report["structure"]={}
        for name,model in models.items():
            structures[name],report["structure"][name]=structure_panel(model,truth,cfg,373017)
        report["structure_change"]=paired_structure_change(structures["broad"],structures["baseline"],structure_target(truth),373018)
        m=report["structure"]["broad"]
        report["low_order_gate"]=m["marginal_mean_absmax"]<.05 and m["marginal_variance_error_absmax"]<.15 and m["pair_covariance_absmax"]<.05
        report["coverage_gate"]=report["paired_coverage_change"]["any_k8"]["lo95"]>0
        report.update(state="completed",best_epoch=fit["best_epoch"],best_step=fit["best_step"],
            decision="coverage_improved" if report["low_order_gate"] and report["coverage_gate"] else "coverage_not_established")
    except BaseException as exc:
        report.update(state="failed_or_interrupted",error=repr(exc));raise
    finally:
        atomic_json(output/"report.json",report)
    print({k:v for k,v in report.items() if k not in ('structure','config')},flush=True)
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("artifacts/dgpo_toy"))
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(1);run(**vars(args))
