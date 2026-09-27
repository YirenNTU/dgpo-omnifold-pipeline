"""Read-only learned-vs-analytic denoiser audit. Never substitutes oracle for RL."""
import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
from .conditional import Config, Denoiser, alpha_sigma
from .parity_cube import CubeDistribution, generation_panel, baseline_gate
from .truth_pretrain import atomic_json


class ExactCubeVelocity(torch.nn.Module):
    """Conditional mean VP velocity of the declared Gaussian-mixture data."""
    def __init__(self,data):
        super().__init__();self.data=data

    def forward(self,x,t,c):
        t=t.expand(x.shape[:-1]);c=c.expand(*x.shape[:-1],1)
        a,s=alpha_sigma(t)
        variance=a.square()*self.data.width**2+s.square()
        centers=self.data.centers.to(x)
        if self.data.continuous:
            priors=self.data.probabilities(c[...,0]).to(x)
        else:
            good=centers.prod(-1)==c[...,0,None]
            priors=torch.where(good,self.data.mass/4,(1-self.data.mass)/4)
        distance=(x[...,None,:]-a[...,None,None]*centers).square().sum(-1)
        posterior=(priors.log()-distance/(2*variance[...,None])).softmax(-1)
        mean_center=posterior@centers
        return s[...,None]/variance[...,None]*(a[...,None]*(1-self.data.width**2)*x-mean_center)


def run(source,output,steps=(50,100,200),events=4096):
    saved=torch.load(source/"best_pretrain.pt",map_location="cpu",weights_only=True)
    cfg=replace(Config(**saved["config"]),eval_events=events)
    report_source=json.loads((source/"report.json").read_text())
    data=CubeDistribution(cfg,report_source["reference_mass"],report_source["width"],
        continuous=saved.get("continuous_condition",False),reference_sharpness=report_source.get("reference_sharpness",1.),condition_frequency=report_source.get("condition_frequency"))
    learned=Denoiser(cfg);learned.load_state_dict(saved["model"]);learned.eval()
    oracle=ExactCubeVelocity(data).eval()
    output.mkdir(parents=True,exist_ok=False)
    report={"state":"running","source":str(source.resolve()),"selected_epoch":saved["epoch"],
        "seed":680017,"events":events,"candidates_per_condition":cfg.candidates,
        "scope":"Read-only sampler/pretrain localization; analytic velocity is NOT a trained diffusion or DGPO result.",
        "arms":{},"decision":"Compare learned50/100/200 and analytic50/100/200 under unchanged generation gate; no automatic model replacement or RL launch."}
    atomic_json(output/"report.json",report)
    try:
        with (output/"progress.jsonl").open("w") as log:
            for name,model in (("learned",learned),("analytic",oracle)):
                for count in steps:
                    _,_,stats=generation_panel(model,data,replace(cfg,ddim_steps=count),680017)
                    row={"model":name,"ddim_steps":count,"gate_pass":baseline_gate(stats),"metrics":stats}
                    report["arms"][f"{name}_{count}"]=row
                    line=json.dumps(row,allow_nan=False);log.write(line+"\n");log.flush();print(line,flush=True)
                    atomic_json(output/"report.json",report)
        report["state"]="completed"
    except BaseException as exc:
        report.update(state="failed_or_interrupted",error=repr(exc));raise
    finally:atomic_json(output/"report.json",report)
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,default=Path("artifacts/dgpo_toy/parity_cube_continuous_4x_v1"))
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--events",type=int,default=4096)
    args=p.parse_args();torch.set_num_threads(1);run(**vars(args))
