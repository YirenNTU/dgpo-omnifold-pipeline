"""Matched finite-K pilot. No refit, no external proposal, no oracle training reward."""
import argparse
import copy
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch
from .conditional import Config, Distribution, Denoiser, generator, dgpo_objective, paired_gain
from .low_ess_recovery import JointFourierClassifier
from .direct_transport import FixedTransport, physical_panel
from .broad_coverage import region_hits
from .structure_metrics import structure_target, paired_structure_change
from .truth_pretrain import atomic_json, atomic_checkpoint

ARMS = {"A_k8": (8, 1), "B_k128": (128, 1), "C_16xk8": (8, 16)}


class ObservedReward(torch.nn.Module):
    """Record coverage of the exact rollout passed into the unchanged loss."""
    def __init__(self, critic, transport):
        super().__init__()
        self.critic, self.transport = critic, transport
        self.hits = []

    def forward(self, x, c, data):
        y = self.transport(x, c)
        self.hits.append(region_hits(y, c, self.transport.data).float())
        return self.critic(y, c, self.transport.data)


def draw_pool(data, cfg, seed):
    rng = generator(seed)
    c = data.contexts(cfg.batch, rng)
    # Same contexts, 16 independent blocks of eight noise candidates.
    noise = torch.randn(cfg.batch, 16, 8, cfg.dimensions, generator=rng)
    t = .7 * torch.rand(16, cfg.timesteps, cfg.batch, generator=rng)
    eps = torch.randn(16, cfg.timesteps, cfg.batch, cfg.dimensions, generator=rng)
    return c, noise, t, eps


def loss_inputs(pool, arm):
    c, noise, t, eps = pool
    if arm == "B_k128":
        yield c, noise.flatten(1, 2), t[0], eps[0]
    else:
        for i in range(ARMS[arm][1]):
            yield c, noise[:, i], t[i], eps[i]


def vector_grads(loss, parameters):
    return torch.cat([g.detach().flatten().double() for g in
        torch.autograd.grad(loss, parameters, retain_graph=True)])


def gradient_summary(g, h):
    ng, nh = float(g.norm()), float(h.norm())
    return {"main_gradient_norm": ng, "velocity_gradient_norm": nh,
        "velocity_to_main_gradient_ratio": nh/ng if ng else None,
        "main_velocity_gradient_cosine": float(g@h)/(ng*nh) if ng*nh else None,
        "total_on_main_projection": float(g@(g+h))/(ng*ng) if ng else None,
        "component_cancellation_ratio": float((g+h).norm())/(ng+nh) if ng+nh else None}


def accumulate(model, reference, reward, data, cfg, pool, arm, trace=False):
    """Each call completes its own nonlinear gate BEFORE averaging gradients."""
    reward.hits.clear()
    count = ARMS[arm][1]
    parameters = tuple(model.parameters())
    parts = []
    def capture(main, penalty):
        parts.append((vector_grads(main, parameters)/count,
                      vector_grads(penalty, parameters)/count))
    diagnostics = []
    for c, noise, t, eps in loss_inputs(pool, arm):
        loss, row = dgpo_objective(model, reference, reward, data, cfg, c, noise, t, eps,
            velocity_coefficient=1., component_callback=capture if trace else None)
        (loss/count).backward()
        diagnostics.append(row)
    # These are explicitly means of call-level summaries, not pooled ESS/cosines.
    result = {"call_mean/"+key: sum(row[key] for row in diagnostics)/count
        for key in diagnostics[0] if all(isinstance(row[key], (int,float)) for row in diagnostics)}
    hits = torch.cat(reward.hits, dim=1)
    result.update(candidate_hit_fraction=float(hits.mean()),
        loss_group_hit_fraction=sum(float(h.amax(1).mean()) for h in reward.hits)/count,
        update_pool_hit_fraction=float(hits.amax(1).mean()),
        good_candidates_per_condition=float(hits.sum(1).mean()),
        candidates_per_update=hits.numel(), loss_calls=count)
    vectors = None
    if trace:
        g = torch.stack([p[0] for p in parts]).sum(0)
        h = torch.stack([p[1] for p in parts]).sum(0)
        result.update(gradient_summary(g,h))
        vectors = {"main": g, "velocity": h, "total": g+h}
    return result, vectors


def flatten_metrics(values, prefix=""):
    result={}
    for key,value in values.items():
        name=prefix+key
        if isinstance(value,dict):result.update(flatten_metrics(value,name+"/"))
        elif isinstance(value,(int,float,bool)):result[name]=value
    return result


def restore_arms(directory, models, opts, cfg, source, amplitude, seed):
    states={arm:torch.load(directory/(arm+".pt"),map_location="cpu",weights_only=True) for arm in ARMS}
    steps={s["step"] for s in states.values()}
    if len(steps)!=1 or not 0<next(iter(steps))<cfg.policy_steps:
        raise ValueError("All arms must resume from the same step before the new horizon")
    for arm,s in states.items():
        if (s["arm"]!=arm or s["seed"]!=seed or s["amplitude"]!=amplitude
            or s["velocity_coefficient"]!=1. or Path(s["source"]).resolve()!=source.resolve()):
            raise ValueError("Resume must preserve source, arm, seed, transport and reference coefficient")
        for key,value in asdict(cfg).items():
            if key not in ("policy_steps","eval_every","eval_events") and s["config"][key]!=value:
                raise ValueError("Resume changed training config: "+key)
        models[arm].load_state_dict(s["model"])
        opts[arm].load_state_dict(s["optimizer"])
    return next(iter(steps))


def run(source, output, steps=100, eval_every=25, eval_events=4096, seed=17, wandb_mode="offline",
        resume=None, endpoint_seed=530017):
    if min(steps, eval_every, eval_events) < 1:
        raise ValueError("Positive budgets required")
    saved = torch.load(source/"reward.pt", map_location="cpu", weights_only=True)
    if saved["feature_mode"] != "joint3":
        raise ValueError("Expected the frozen strong joint3 classifier")
    cfg = replace(Config(**saved["config"]), policy_steps=steps,
                  eval_every=eval_every, eval_events=eval_events)
    data = Distribution(cfg)
    initial = Denoiser(cfg); initial.load_state_dict(saved["initial"]); initial.eval()
    reference = copy.deepcopy(initial).requires_grad_(False)
    critic = JointFourierClassifier(cfg); critic.load_state_dict(saved["classifier"])
    critic.eval().requires_grad_(False)
    transform = FixedTransport(data, saved["amplitude"])
    output.mkdir(parents=True, exist_ok=False)
    report = {"state":"running", "source":str((source/"reward.pt").resolve()),
        "source_policy_step":0, "classifier":"frozen joint3; no refit",
        "config":asdict(cfg), "amplitude":saved["amplitude"], "velocity_coefficient":1.,
        "seeds":{"train_base":seed+510000,"monitor":520017,"endpoint":endpoint_seed},
        "resume_from":str(resume.resolve()) if resume else None,
        "scope":"Finite-K/grouping versus within-condition gradient averaging, not pure coverage causality.",
        "matching":"B/C same candidate noise pool, conditions, candidate-loss evaluations; C has independent t/eps per K8 call; B broadcasts one t/eps per condition as native loss does.",
        "primary":"Independent fixed-reward gain at final step; paired B-A and B-C contrasts. Structure and low-order errors are separate checks.",
        "decision":"B must beat both A and C with paired 95% CI above zero, show higher loss-group coverage, and pass structural checks before supporting a group-coverage/finite-K mechanism. No automatic extension.",
        "arms":{}}
    atomic_json(output/"report.json", report)
    import wandb
    wb=wandb.init(project="dgpo-toy", mode=wandb_mode, dir=str(output.resolve()),
        name="Does rollout coverage improve learning? | frozen strong toy | K8 vs K128 vs 16xK8",
        group="Finite rollout coverage", config=report,
        tags=["toy","frozen-strong","velocity-mse-1","matched-candidate-budget"])
    report["wandb"]={"id":wb.id,"mode":wandb_mode,"directory":wb.dir}
    for arm in ARMS:
        wb.define_metric(arm+"/step")
        wb.define_metric(arm+"/*",step_metric=arm+"/step")
    models = {arm:copy.deepcopy(initial).requires_grad_(True) for arm in ARMS}
    opts = {arm:torch.optim.AdamW(m.parameters(), lr=cfg.policy_lr, weight_decay=cfg.weight_decay)
            for arm,m in models.items()}
    rewards = {arm:ObservedReward(critic, transform) for arm in ARMS}
    start_step=restore_arms(resume,models,opts,cfg,source/"reward.pt",saved["amplitude"],seed) if resume else 0
    report["start_step"]=start_step
    monitor_base = physical_panel(initial,critic,transform,cfg,520017,k=8)
    endpoint_base = physical_panel(initial,critic,transform,cfg,endpoint_seed,k=8)
    report["initial"] = endpoint_base[3]
    times = {arm:0. for arm in ARMS}
    first_gradients = {}
    if resume:
        first_gradients=torch.load(resume/"initial_gradients.pt",map_location="cpu",weights_only=True)
        previous=json.loads((resume/"report.json").read_text())
        times={arm:previous.get("arms",{}).get(arm,{}).get("training_seconds",0.) for arm in ARMS}
        atomic_checkpoint(output/"initial_gradients.pt",first_gradients)
    try:
        with (output/"progress.jsonl").open("w") as log:
            def emit(row):
                line = json.dumps(row,allow_nan=False)
                log.write(line+"\n");log.flush()
                if "monitor" in row:print(line,flush=True)
                wb.log(flatten_metrics(row,row["arm"]+"/"))
            for step in range(start_step+1,steps+1):
                pool = draw_pool(data,cfg,seed+510000+step)
                trace = step==start_step+1 or step%eval_every==0 or step==steps
                for arm,model in models.items():
                    started = time.monotonic()
                    opt=opts[arm];opt.zero_grad(set_to_none=True)
                    row, vectors = accumulate(model,reference,rewards[arm],data,cfg,pool,arm,trace)
                    if step==1:first_gradients[arm]=vectors
                    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
                    before = [p.detach().clone() for p in model.parameters()]
                    opt.step()
                    delta = sum(float((p.detach()-old).square().sum()) for p,old in zip(model.parameters(),before))**.5
                    times[arm] += time.monotonic()-started
                    row.update(phase="update",arm=arm,step=step,gradient_norm=norm,
                        grad_clipped=norm>1,update_l2=delta,training_seconds=times[arm],
                        cumulative_candidates=step*cfg.batch*ARMS[arm][0]*ARMS[arm][1],
                        cumulative_candidate_time_evaluations=step*cfg.batch*ARMS[arm][0]*ARMS[arm][1]*cfg.timesteps)
                    if trace:
                        r,f,_,summary=physical_panel(model,critic,transform,cfg,520017,k=8)
                        row.update(monitor=summary,monitor_gain=paired_gain(r,monitor_base[0]),
                            monitor_structure_change=paired_structure_change(f,monitor_base[1],structure_target(data),520018))
                        atomic_checkpoint(output/(arm+".pt"),{"model":model.state_dict(),
                            "optimizer":opt.state_dict(),"step":step,"arm":arm,"config":asdict(cfg),
                            "amplitude":saved["amplitude"],"velocity_coefficient":1.,"seed":seed,
                            "source":report["source"]})
                    emit(row)
                report["completed_steps"]=step
                if step==1:atomic_checkpoint(output/"initial_gradients.pt",first_gradients)
                atomic_json(output/"report.json",report)
            endpoints = {}
            for arm,model in models.items():
                r,f,_,summary=physical_panel(model,critic,transform,cfg,endpoint_seed,k=8)
                endpoints[arm]=(r,f)
                report["arms"][arm]={"endpoint":summary,"reward_gain":paired_gain(r,endpoint_base[0]),
                    "structure_change":paired_structure_change(f,endpoint_base[1],structure_target(data),530018),
                    "training_seconds":times[arm]}
            report["paired_comparisons"]={}
            for other in ("A_k8","C_16xk8"):
                report["paired_comparisons"]["B_minus_"+other]={
                    "reward":paired_gain(endpoints["B_k128"][0],endpoints[other][0]),
                    "structure":paired_structure_change(endpoints["B_k128"][1],endpoints[other][1],structure_target(data),530019)}
            report["initial_gradient_cosines"]={}
            for other in ("A_k8","C_16xk8"):
                g=first_gradients["B_k128"]["main"];h=first_gradients[other]["main"]
                report["initial_gradient_cosines"]["B_vs_"+other]=float(g@h/(g.norm()*h.norm()))
            atomic_checkpoint(output/"initial_gradients.pt",first_gradients)
            report["frozen_sources_unchanged"] = all(torch.equal(v,saved["classifier"][k]) for k,v in critic.state_dict().items()) and all(torch.equal(v,saved["initial"][k]) for k,v in reference.state_dict().items())
            report["state"]="completed"
    except BaseException as exc:
        report.update(state="failed_or_interrupted",error=repr(exc));raise
    finally:
        atomic_json(output/"report.json",report)
        wb.summary.update(flatten_metrics(report))
        wb.summary["state"]=report["state"]
        wb.finish(exit_code=0 if report["state"]=="completed" else 1)
    return report


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,default=Path("artifacts/dgpo_toy/direct_transport_joint_v1"))
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--steps",type=int,default=100)
    p.add_argument("--eval-every",type=int,default=25)
    p.add_argument("--eval-events",type=int,default=4096)
    p.add_argument("--seed",type=int,default=17)
    p.add_argument("--wandb-mode",choices=("offline","online","disabled"),default="offline")
    p.add_argument("--resume",type=Path)
    p.add_argument("--endpoint-seed",type=int,default=530017)
    args=p.parse_args();torch.set_num_threads(1);run(**vars(args))
