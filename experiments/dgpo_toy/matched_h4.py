"""Fit the existing joint3 H4 on exactly the diffusion's fixed truth rows."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from .conditional import Config, Distribution, Denoiser, Classifier, generator, initialize, ddim, classifier_metrics
from .low_ess_recovery import JointFourierClassifier
from .truth_pretrain import atomic_json, atomic_checkpoint, early_stop_update


def load_inputs(dataset_path, diffusion_path, policy_source=False):
    dataset = torch.load(dataset_path, map_location="cpu", weights_only=True)
    state = torch.load(diffusion_path, map_location="cpu", weights_only=True)
    evidence = json.loads(diffusion_path.with_name("report.json").read_text())
    cfg = Config(**state["config"])
    if policy_source:
        suite = json.loads(diffusion_path.parent.parent.joinpath("report.json").read_text())
        if (suite["state"] != "completed" or state["arm"] != "strong_no_kl"
                or state["step"] != 10000 or state["velocity_coefficient"] != 0.
                or suite["config"] != state["config"]
                or suite["arms"]["strong_no_kl"]["steps"] != state["step"]
                or Path(suite["dataset"]).resolve() != dataset_path.resolve()
                or suite["source"] != state["source"]):
            raise ValueError("Requires completed raw pure-strong 10k policy source")
        load_inputs(dataset_path, Path(state["source"]))
        model = Denoiser(cfg)
        model.load_state_dict(state["model"], strict=True)
        model.eval().requires_grad_(False)
        return cfg, dataset, model, state
    if (state.get("weights") != "raw" or state.get("trained_on") != "fixed_complete_truth_dataset"
            or evidence["state"] != "completed" or evidence["config"] != state["config"]
            or evidence["best_step"] != state["step"] or evidence["best_epoch"] != state["epoch"]
            or Path(evidence["dataset"]).resolve() != dataset_path.resolve()
            or evidence["dataset_metadata"] != dataset["metadata"]):
        raise ValueError("Requires the selected raw truth-trained diffusion and its exact dataset")
    model = Denoiser(cfg)
    model.load_state_dict(state["model"])
    model.eval().requires_grad_(False)
    return cfg, dataset, model, state


@torch.no_grad()
def paired_panels(dataset, model, cfg, seed):
    panels = {}
    for i, name in enumerate(("train", "validation", "test")):
        split = dataset[name]
        c = split["condition"]
        noise = torch.randn(len(c), cfg.dimensions, generator=generator(seed+70000+1000*i))
        generated = torch.cat([ddim(model, cc, z, cfg.ddim_steps)
            for cc, z in zip(c.split(512), noise.split(512))])
        if not torch.isfinite(generated).all():
            raise FloatingPointError("Nonfinite diffusion samples")
        panels[name] = {"context": c, "truth": split["target"], "generated": generated}
    return panels


def run(output, *, dataset_path, diffusion_path, seed=17, patience=20, min_delta=1e-4,
        resume_from=None, stop_after_epoch=None, feature_mode="joint3", panels_from=None,
        policy_source=False):
    if patience < 1 or not math.isfinite(min_delta) or min_delta < 0:
        raise ValueError("Positive patience and finite nonnegative min_delta required")
    if feature_mode not in ("joint3", "plain"):
        raise ValueError("Expected joint3 or plain classifier")
    if output.resolve() in (dataset_path.parent.resolve(), diffusion_path.parent.resolve()):
        raise ValueError("Separate output required to preserve input artifacts")
    if panels_from is not None and output.resolve() == panels_from.parent.resolve():
        raise ValueError("Separate output required to preserve shared-panel classifier")
    cfg, dataset, diffusion, source = load_inputs(dataset_path, diffusion_path, policy_source)
    if cfg.fit_batch < 2 or cfg.fit_batch % 2:
        raise ValueError("Balanced classifier batches require an even fit_batch")
    data = Distribution(cfg)
    def new_classifier():
        return (initialize(JointFourierClassifier,cfg,seed+4000) if feature_mode == "joint3"
                else initialize(Classifier,cfg,seed+4000,False))
    model = new_classifier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.classifier_lr, weight_decay=cfg.weight_decay)
    rng = generator(seed+81000)
    contract = {"experiment": "same_truth_fixed_h4_v1", "seed": seed,
        "dataset": str(dataset_path.resolve()), "diffusion_checkpoint": str(diffusion_path.resolve()),
        "patience_epochs": patience, "min_delta_absolute": min_delta, "feature_mode": feature_mode,
        "selection": "minimum validation BCE", "weights": "raw_no_ema"}
    if panels_from is not None:
        contract["panels_from"] = str(panels_from.resolve())
    if policy_source:
        contract["source_kind"] = "raw_pure_strong_10k_policy"
    report = {**contract, "config": asdict(cfg), "state": "preparing", "history": [],
        "source_diffusion_step": source["step"], "source_diffusion_epoch": source.get("epoch"),
        "completed_epochs": 0, "completed_steps": 0, "best_step": 0, "best_epoch": 0,
        "max_steps": None, "max_epochs": None, "new_truth_events": 0,
        "dataset_metadata": dataset["metadata"], "parameters": sum(p.numel() for p in model.parameters()),
        "negative_sampling_seeds": {name: seed+70000+1000*i for i,name in enumerate(("train","validation","test"))},
        "caveat": "Same truth rows, not architecture/capacity/feature-prior or compute matching"}
    epoch, step, stale, anchor, best_loss = 0, 0, 0, float("inf"), float("inf")
    best_model, standardized = copy.deepcopy(model.state_dict()), False
    if resume_from is not None:
        if resume_from.parent.resolve() == output.resolve():
            raise ValueError("Resume into a separate output")
        state = torch.load(resume_from, map_location="cpu", weights_only=True)
        if state["contract"] != contract or state["config"] != asdict(cfg):
            raise ValueError("Resume contract mismatch")
        if state["report"]["state"] == "completed":
            raise ValueError("Classifier already early-stopped")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        rng.set_state(state["rng"])
        epoch, step, stale, anchor = state["epoch"], state["step"], state["stale"], state["anchor"]
        best_model, best_loss, standardized = state["best_model"], state["best_loss"], state["standardized"]
        report = state["report"]
        report["resumed_from"] = str(resume_from.resolve())
        panels = torch.load(resume_from.parent/"panels.pt", map_location="cpu", weights_only=True)
    elif panels_from is not None:
        panel_report = json.loads(panels_from.with_name("report.json").read_text())
        if (Path(panel_report["dataset"]).resolve() != dataset_path.resolve()
                or Path(panel_report["diffusion_checkpoint"]).resolve() != diffusion_path.resolve()
                or panel_report["state"] != "completed"):
            raise ValueError("Shared panels must come from the same dataset and diffusion")
        panels = torch.load(panels_from,map_location="cpu",weights_only=True)
        report["negative_sampling_seeds"] = panel_report["negative_sampling_seeds"]
    else:
        panels = paired_panels(dataset, diffusion, cfg, seed)
    for name, panel in panels.items():
        if not torch.equal(panel["context"], dataset[name]["condition"]) or not torch.equal(panel["truth"], dataset[name]["target"]):
            raise ValueError("Classifier truth rows must match the diffusion dataset exactly")
    report["splits"] = {name: {"truth":len(p["truth"]), "generated":len(p["generated"]),
        "contexts":len(p["context"]), "exact_truth_and_context_match":True} for name,p in panels.items()}
    output.mkdir(parents=True, exist_ok=True)
    atomic_checkpoint(output/"panels.pt", panels)
    started, previous_seconds = time.perf_counter(), report.get("seconds", 0.)
    report["state"] = "training"

    def save():
        report.update(completed_steps=step, completed_epochs=epoch, stale_epochs=stale,
                      seconds=previous_seconds+time.perf_counter()-started)
        atomic_checkpoint(output/"last_state.pt", {"contract":contract, "config":asdict(cfg),
            "model":model.state_dict(), "optimizer":optimizer.state_dict(), "rng":rng.get_state(),
            "epoch":epoch, "step":step, "stale":stale, "anchor":anchor, "best_model":best_model,
            "best_loss":best_loss, "standardized":standardized, "report":report})
        atomic_json(output/"report.json", report)

    def export_best():
        atomic_checkpoint(output/"best_classifier.pt", {"config":asdict(cfg), "classifier":best_model,
            "feature_mode":feature_mode, "step":report["best_step"], "epoch":report["best_epoch"],
            "seed":seed, "validation_bce":best_loss, "source":str(diffusion_path.resolve()),
            "dataset":str(dataset_path.resolve()), "weights":"raw"})

    with (output/"progress.jsonl").open("w") as log:
        def emit(row):
            line = json.dumps(row, allow_nan=False)
            print(line, flush=True)
            log.write(line+"\n")
            log.flush()
        emit({"phase":"start", **contract, "splits":report["splits"]})
        save()
        try:
            train = panels["train"]
            while stale < patience:
                epoch += 1
                total, max_grad, clipped = 0., 0., 0
                order = torch.randperm(len(train["context"]), generator=rng)
                for ids in order.split(cfg.fit_batch//2):
                    step += 1
                    if step == cfg.standardize_step and not standardized:
                        count = min(2048, len(train["context"]))
                        model.standardize(torch.cat([train["truth"][:count],train["generated"][:count]]),
                            train["context"][:count].repeat(2,1), data)
                        for p in model.head.parameters():
                            optimizer.state.pop(p, None)
                        standardized = True
                    c = train["context"][ids].repeat(2,1)
                    y = torch.cat([train["truth"][ids],train["generated"][ids]])
                    labels = torch.cat([torch.ones(len(ids)),torch.zeros(len(ids))])
                    loss = F.binary_cross_entropy_with_logits(model(y,c,data),labels)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite classifier loss")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
                    optimizer.step()
                    total += float(loss.detach())*len(ids)
                    max_grad, clipped = max(max_grad,norm),clipped+int(norm>1.)
                val = classifier_metrics(model,panels["validation"],data)
                if not all(math.isfinite(x) for x in val.values()):
                    raise FloatingPointError("Nonfinite validation metrics")
                if val["bce"] < best_loss:
                    best_loss,best_model = val["bce"],copy.deepcopy(model.state_dict())
                    report.update(best_epoch=epoch,best_step=step,best_validation_bce=best_loss)
                    export_best()
                anchor,stale = early_stop_update(val["bce"],anchor,stale,min_delta)
                row = {"phase":"epoch","epoch":epoch,"step":step,"train_bce":total/len(order),
                    "validation":val,"max_gradient_norm":max_grad,"clipped_updates":clipped,
                    "stale_epochs":stale,"best_epoch":report["best_epoch"],"lr":cfg.classifier_lr}
                report["history"].append(row)
                emit(row)
                save()
                if epoch == stop_after_epoch and stale < patience:
                    report["state"]="paused"
                    save()
                    return report
            report.update(state="evaluating",stop_reason="validation_early_stopping")
            save()
            selected = new_classifier()
            selected.load_state_dict(best_model)
            selected.eval().requires_grad_(False)
            report["selected_metrics"] = {name:classifier_metrics(selected,panel,data) for name,panel in panels.items()}
            report["source_unchanged"] = all(torch.equal(v,source["model"][k]) for k,v in diffusion.state_dict().items())
            report["state"]="completed"
            export_best()
            save()
            emit({"phase":"complete","epoch":epoch,"step":step,"best_epoch":report["best_epoch"],
                  "test":report["selected_metrics"]["test"],"seconds":report["seconds"]})
            return report
        except BaseException as exc:
            atomic_json(output/"error.json",{"type":type(exc).__name__,"message":str(exc),
                "attempted_epoch":epoch,"resume":"last_state.pt: last complete saved epoch"})
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",required=True,type=Path)
    parser.add_argument("--diffusion-checkpoint",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--seed",type=int,default=17)
    parser.add_argument("--patience",type=int,default=20)
    parser.add_argument("--min-delta",type=float,default=1e-4)
    parser.add_argument("--resume-from",type=Path)
    parser.add_argument("--feature-mode",choices=("joint3","plain"),default="joint3")
    parser.add_argument("--panels-from",type=Path)
    args=parser.parse_args()
    torch.set_num_threads(1)
    run(args.output,dataset_path=args.dataset,diffusion_path=args.diffusion_checkpoint,
        seed=args.seed,patience=args.patience,min_delta=args.min_delta,resume_from=args.resume_from,
        feature_mode=args.feature_mode,panels_from=args.panels_from)


if __name__=="__main__":
    main()
