"""Saved no-Fourier classifier DGPO ablation; no classifier fitting."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import torch

from experiments.dgpo_toy.conditional import (
    Config, Classifier, Distribution, Denoiser, generator, ddim, make_panel, policy_train,
    policy_panel, paired_gain, scores, weight_health, classifier_metrics,
)
from experiments.dgpo_toy.low_ess_recovery import confirmation_metrics
from experiments.dgpo_toy.fixed_reward import load_source, unchanged, REQUIRED_GATES, WAIVED_GATES
from experiments.dgpo_toy.velocity_kl import load_comparator
from experiments.dgpo_toy.direct_density import EndpointRatio, DensitySettings, DensityValidityError
from experiments.dgpo_toy.net_reward_kl import density_panel
from experiments.dgpo_toy.structure_metrics import structure_panel, structure_target, paired_structure_change


@torch.no_grad()
def paired_scores(model, weak, strong, data, cfg, seed):
    rng = generator(seed)
    c = data.contexts(cfg.eval_events, rng)
    noise = torch.randn(cfg.eval_events, cfg.candidates, cfg.dimensions, generator=rng)
    outputs = {"weak": [], "strong": []}
    for cc, zz in zip(c.split(128), noise.split(128)):
        y = ddim(model, cc[:, None], zz, cfg.ddim_steps)
        ctx = cc[:, None].expand(-1, cfg.candidates, -1)
        for name, critic in (("weak", weak), ("strong", strong)):
            outputs[name].append(critic(y, ctx, data))
    return {k: torch.cat(v) for k, v in outputs.items()}


def signal_metrics(panels):
    result, centered = {}, {}
    for name, values in panels.items():
        values = values.double()
        centered[name] = values-values.mean(1, keepdim=True)
        result[name] = {
            **weight_health(values), "reward_mean": float(values.mean()),
            "reward_std": float(values.std(unbiased=False)),
            "within_k_ess_fraction": float((1/(values.shape[1]*values.softmax(1).square().sum(1))).mean()),
            "centered_reward_rms": float(centered[name].square().mean().sqrt()),
            "informative_group_fraction": float(((values.max(1).values-values.min(1).values)>1e-3).double().mean()),
        }
    a, b = centered["weak"], centered["strong"]
    norm = a.norm()*b.norm()
    result["centered_weak_strong_cosine"] = float((a*b).sum()/norm) if norm > 0 else 0.
    result["candidate_sign_agreement"] = float(((a*b)>0).double().mean())
    return result


def setup_gates(confirmation, signal, checkpoint_verified):
    weak, strong = confirmation["weak"], confirmation["strong"]
    return {
        "saved_plain_checkpoint_verified": checkpoint_verified,
        "weaker_but_informative": .70 <= weak["auc"] < strong["auc"] and weak["bce"] <= .60,
        "higher_ess": .05 <= weak["ess_fraction"] <= .80 and weak["ess_fraction"] >= 10*strong["ess_fraction"],
        "within_context_alignment": signal["centered_weak_strong_cosine"] > .10,
        "within_context_signal": signal["weak"]["informative_group_fraction"] >= .01,
    }


def load_plain(path, source_payload, cfg, seed):
    """Load the validation-selected plain model, never retrain or alter its logits."""
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    evidence = json.loads(path.with_name("report.json").read_text())
    if bundle["config"] != evidence["config"] or evidence["seed"] != seed:
        raise ValueError("Plain checkpoint/report config or seed mismatch")
    # Classifier fit budget differs by design; the policy horizon is set by this run.
    expected = {**bundle["config"], "classifier_steps": cfg.classifier_steps,
                "policy_steps": cfg.policy_steps}
    if expected != asdict(cfg):
        raise ValueError("Plain checkpoint must use the same distribution and model setup")
    if (bundle["initial"].keys() != source_payload["initial"].keys() or
            not all(torch.equal(v, source_payload["initial"][k]) for k, v in bundle["initial"].items())):
        raise ValueError("Plain and strong rewards must share exactly the same pretrained generator")
    fit = evidence["classifiers"]["plain"]
    selected = [r for r in fit["history"] if r["step"] == fit["selected_step"]]
    if len(selected) != 1 or selected[0]["bce"] != fit["validation_bce"]:
        raise ValueError("Plain validation selection is inconsistent")
    model = Classifier(Config(**bundle["config"]), False)
    model.load_state_dict(bundle["classifiers"]["plain"], strict=True)
    model.eval().requires_grad_(False)
    if not all(torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError("Nonfinite plain classifier checkpoint")
    return model, bundle, fit, selected[0]


def load_control(directory, source, cfg, seed, regularization):
    if regularization == "none":
        return load_comparator(directory, source, cfg, seed)
    report = json.loads((directory/"report.json").read_text())
    state = torch.load(directory/"dgpo.pt", map_location="cpu", weights_only=True)
    if (report["config"] != asdict(cfg) or state["config"] != asdict(cfg)
            or state["step"] != cfg.policy_steps
            or report["seed"] != seed
            or report["stream_seeds"]["train"] != seed+3000
            or report["stream_seeds"]["monitor"] != seed+90000
            or Path(report["source"]).resolve() != source.resolve()
            or Path(state["source"]).resolve() != source.resolve()
            or not report["frozen_source_unchanged"]):
        raise ValueError("Control must match source, budget, config and seed")
    if regularization == "velocity":
        if (report.get("experiment") != "conditional_velocity_mse_ablation_v1"
                or report.get("decision") not in ("reward_improved_by_velocity", "reward_suppressed_by_velocity", "unresolved")
                or state["arm"] != "dgpo" or state.get("velocity_coefficient") != 1.
                or report.get("velocity_coefficient") != 1.):
            raise ValueError("Velocity control must be completed with velocity-MSE coefficient1")
        name = "velocity_kl"
    elif regularization == "endpoint":
        if (report.get("state") != "completed" or report["completed_steps"] != cfg.policy_steps
                or state["arm"] != "net_reward_kl"
                or report["density_settings"] != asdict(DensitySettings())):
            raise ValueError("Endpoint control must be completed with direct coefficient1 settings")
        name = "net_reward_kl"
    else:
        raise ValueError("Unknown regularization")
    history = report["policy_histories"][name]
    if len(history) != cfg.policy_steps or any(x["step"] != i+1 for i, x in enumerate(history)):
        raise ValueError("Incomplete control history")
    model = Denoiser(cfg).eval().requires_grad_(False)
    model.load_state_dict(state["model"])
    return model, report


def endpoint_decision(endpoints, comparisons, regularization):
    weak = endpoints["weak_policy"]
    structure = weak["structure"]
    checks = {
        "common_reward_transfer": weak["strong_gain"]["gain"] >= .10 and weak["strong_gain"]["lo95"] > 0,
        "beats_strong_policy_reward": comparisons["strong_reward"]["lo95"] > 0,
        "high_order_vs_initial": weak["structure_change"]["hi95"] < 0 and
            structure["fourier_moment_rmse"] <= .9*endpoints["initial"]["structure"]["fourier_moment_rmse"],
        "high_order_vs_control": comparisons["structure"]["hi95"] < 0,
        "lower_order_preserved": structure["marginal_mean_absmax"] <= .10 and
            structure["marginal_variance_error_absmax"] <= .15 and structure["pair_covariance_absmax"] <= .08,
    }
    transfer = checks["common_reward_transfer"] and checks["beats_strong_policy_reward"]
    joint = transfer and checks["high_order_vs_initial"] and checks["high_order_vs_control"]
    useful = joint and (regularization == "none" or checks["lower_order_preserved"])
    return ("supports_weak_signal_structural_progress" if useful else
            "supports_weak_signal_reward_transfer" if transfer else "no_decisive_weak_signal_advantage"), checks


def run(source, comparator, output, *, plain_checkpoint, regularization="endpoint", policy_steps=10000,
        seed=17, smoke=False):
    if regularization not in ("endpoint", "velocity", "none") or policy_steps < 1:
        raise ValueError("Valid regularization and positive update budgets required")
    if output.resolve() in (source.parent.resolve(), comparator.resolve(), plain_checkpoint.parent.resolve()):
        raise ValueError("Use separate output to preserve source and control")
    cfg, initial, strong, payload, evidence = load_source(source)
    cfg = replace(cfg, policy_steps=policy_steps)
    if payload["feature_mode"] != "joint3" or not all(evidence["validity"].get(k) is True for k in REQUIRED_GATES):
        raise ValueError("Requires the valid joint3 pretrained source")
    weak, plain_payload, fit, selected = load_plain(plain_checkpoint, payload, cfg, seed)
    control, previous = load_control(comparator, source, cfg, seed, regularization)
    data = Distribution(cfg)
    old_name = {"endpoint": "net_reward_kl", "velocity": "velocity_kl", "none": "dgpo"}[regularization]
    velocity_coefficient = 1. if regularization == "velocity" else 0.
    control_monitor = policy_panel(control, strong, data, cfg, seed+90000)[0]
    if float(control_monitor.mean()) != previous["policy_histories"][old_name][-1]["monitor_reward"]:
        raise ValueError("Saved control failed its original strong-reward monitor")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report = {"experiment": "saved_no_fourier_classifier_v1", "state": "preparing", "decision": "running",
        "source": str(source.resolve()), "comparator": str(comparator.resolve()), "seed": seed,
        "config": asdict(cfg), "regularization": regularization, "smoke": smoke,
        "velocity_coefficient": velocity_coefficient,
        "velocity_proxy": {"formula": "coefficient * 0.5 * mean((v_current-v_step0)^2)",
            "reference": "frozen original pretrained step0", "exact_endpoint_kl": False,
            "sampling": "same detached current-policy candidates, times and noise as DGPO"} if velocity_coefficient else None,
        "plain_checkpoint": str(plain_checkpoint.resolve()),
        "classifier_selected_step": fit["selected_step"], "new_classifier_fit_updates": 0,
        "classifier_fit_budget": plain_payload["config"]["classifier_steps"],
        "strong_classifier_fit_steps": payload["selected_step"],
        "stream_seeds": {"train": seed+3000, "monitor": seed+90000,
            "confirmation": seed+107000, "candidate": seed+108000, "endpoint": seed+109000},
        "density_settings": asdict(DensitySettings()) if regularization == "endpoint" else None,
        "source_ratio_fidelity_waivers": list(WAIVED_GATES), "source_validity": evidence["validity"],
        "completed_steps": 0, "policy_history": [], "endpoints": {},
        "causal_scope": "Saved classifier contrast: architecture, finite-pool vs fresh-data fitting, scale, ranking and approximation differ; not Fourier or ESS alone"}
    def save():
        (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    with (output/"progress.jsonl").open("w") as log:
        def emit(row):
            line = json.dumps({"seed": seed, **row}, allow_nan=False)
            print(line, flush=True)
            log.write(line+"\n")
            log.flush()
        save()
        try:
            plain_cfg = Config(**plain_payload["config"])
            validation = make_panel(initial, data, plain_cfg, plain_cfg.validation_events, seed+20000)
            reproduced = classifier_metrics(weak, validation, data)
            verified = all(abs(reproduced[key]-selected[key]) <= 1e-6 for key in ("bce", "auc", "ess_fraction"))
            report["classifier_provenance"] = {
                "generator_weights_exact": True, "feature_mode": "plain", "fourier_enabled": weak.fourier,
                "parameters": sum(p.numel() for p in weak.parameters()),
                "fit_metadata": {k:v for k,v in fit.items() if k != "history"},
                "original_validation": selected, "reproduced_validation": reproduced,
                "validation_reproduced": verified}
            if not verified:
                raise ValueError("Saved plain weights do not reproduce their selected validation metrics")
            weak_state = copy.deepcopy(weak.state_dict())
            torch.save({"config": asdict(cfg), "initial": payload["initial"], "classifier": weak_state,
                        "source": str(source.resolve()), "selected_step": fit["selected_step"],
                        "feature_mode": "plain", "plain_checkpoint": str(plain_checkpoint.resolve()),
                        "fit_config": plain_payload["config"]}, output/"weak_reward.pt")
            emit({"phase": "confirmation_start", "contexts": 64 if smoke else 131072})
            panel = make_panel(initial, data, cfg, 64 if smoke else 131072, seed+107000)
            confirmation = {}
            for name, critic in (("weak", weak), ("strong", strong)):
                confirmation[name] = confirmation_metrics(critic, panel, data)
                pos, neg = scores(critic, panel, data)
                confirmation[name].update({"truth_logit_std": float(pos.std(unbiased=False)),
                    "generated_logit_std": float(neg.std(unbiased=False))})
            del panel, validation
            report["confirmation"] = confirmation
            candidate = signal_metrics(paired_scores(initial, weak, strong, data, cfg, seed+108000))
            report["candidate_signal"] = candidate
            report["setup_gates"] = setup_gates(confirmation, candidate, verified)
            emit({"phase": "setup_validity", "gates": report["setup_gates"],
                  "confirmation": confirmation, "candidate_signal": candidate})
            save()
            if not smoke and not all(report["setup_gates"].values()):
                report.update(state="stopped_setup", decision="inconclusive_setup", seconds=time.perf_counter()-started)
                save()
                return report
            controller = EndpointRatio() if regularization == "endpoint" else None
            def snapshot(step, model, optimizer, rng, history):
                return {"config": asdict(cfg), "model": copy.deepcopy(model.state_dict()),
                    "optimizer": copy.deepcopy(optimizer.state_dict()), "rng": rng.get_state().clone(),
                    "history": list(history), "step": step, "seed": seed, "arm": "dgpo",
                    "source": str(source.resolve()), "reward_source": str((output/"weak_reward.pt").resolve()),
                    "monitor_seed": seed+90000, "velocity_coefficient": velocity_coefficient,
                    "endpoint_controller": controller.state_dict() if controller else None}
            last_valid = snapshot(0, initial,
                torch.optim.AdamW(initial.parameters(), lr=cfg.policy_lr, weight_decay=cfg.weight_decay),
                generator(seed+3000), [])
            base_scores = paired_scores(initial, weak, strong, data, cfg, seed+90000)
            _, base_structure = structure_panel(initial, data, cfg, seed+90000)
            report["initial_monitor"] = {"signals": signal_metrics(base_scores),
                "structure": {k:v for k,v in base_structure.items() if k not in ("moment_mean", "truth_moment")}}
            def instrument():
                rng = generator(180017)
                return controller.current.validate_autodiff(data.contexts(8, rng),
                    torch.randn(8, cfg.dimensions, generator=rng))
            def checkpoint(step, model, optimizer, rng, history):
                nonlocal last_valid
                measure = step == 1 or step % 250 == 0 or step == cfg.policy_steps
                if controller:
                    try:
                        controller.refresh(model, initial, cfg.ddim_steps)
                        if measure:
                            history[-1].update({"density/autodiff_"+k: v for k, v in instrument().items()})
                    except DensityValidityError:
                        torch.save({"step": step, "model": model.state_dict()}, output/"uncertified_attempt.pt")
                        raise
                if measure:
                    values = paired_scores(model, weak, strong, data, cfg, seed+90000)
                    _, structure = structure_panel(model, data, cfg, seed+90000)
                    row = {"phase": "common_judge_monitor", "step": step,
                        "signals": signal_metrics(values),
                        "strong_gain": paired_gain(values["strong"], base_scores["strong"]),
                        "structure": {k:v for k,v in structure.items() if k not in ("moment_mean", "truth_moment")}}
                    history[-1]["common_judge"] = row
                    if controller:
                        row["density"] = density_panel(controller, model, data, cfg, seed+190000,
                                                       count=min(512, cfg.eval_events))
                    emit(row)
                last_valid = snapshot(step, model, optimizer, rng, history)
                report.update(completed_steps=step, policy_history=history)
                if step == 1 or step % 1000 == 0 or step == cfg.policy_steps:
                    torch.save(last_valid, output/"dgpo_state.pt")
                    save()
            if controller:
                controller.refresh(initial, initial, cfg.ddim_steps)
                report["initial_density_validation"] = {"certificate": controller.current.certificate, "autodiff": instrument()}
            report["state"] = "training"
            save()
            try:
                model, history = policy_train("dgpo", initial, weak, data, cfg, seed, seed+90000,
                    emit, checkpoint, endpoint_controller=controller, velocity_coefficient=velocity_coefficient)
                report["state"] = "completed"
            except DensityValidityError as exc:
                report.update(state="stopped_density_validity", stop=str(exc))
                model = copy.deepcopy(initial)
                model.load_state_dict(last_valid["model"])
                if controller:
                    controller.load_state_dict(last_valid["endpoint_controller"])
                    controller.refresh(model, initial, cfg.ddim_steps)
            torch.save(last_valid, output/"dgpo_state.pt")
            report.update(completed_steps=last_valid["step"], policy_history=last_valid["history"])
            torch.save({"config": asdict(cfg), "model": model.state_dict(), "step": last_valid["step"],
                        "source": str(source.resolve()), "regularization": regularization,
                        "velocity_coefficient": velocity_coefficient}, output/"dgpo.pt")
            panels, structures = {}, {}
            for name, policy in (("initial", initial), ("strong_policy", control), ("weak_policy", model)):
                panels[name] = paired_scores(policy, weak, strong, data, cfg, seed+109000)
                structures[name], summary = structure_panel(policy, data, cfg, seed+109000)
                report["endpoints"][name] = {"step": 0 if name == "initial" else cfg.policy_steps if name == "strong_policy" else last_valid["step"],
                    "signals": signal_metrics(panels[name]), "structure": summary}
            for name in ("strong_policy", "weak_policy"):
                endpoint = report["endpoints"][name]
                endpoint.update({judge+"_gain": paired_gain(panels[name][judge], panels["initial"][judge]) for judge in ("weak", "strong")})
                endpoint["structure_change"] = paired_structure_change(structures[name], structures["initial"], structure_target(data), seed+109001)
            report["comparisons"] = {
                "strong_reward": paired_gain(panels["weak_policy"]["strong"], panels["strong_policy"]["strong"]),
                "structure": paired_structure_change(structures["weak_policy"], structures["strong_policy"], structure_target(data), seed+109002)}
            report["comparison_budgets_matched"] = last_valid["step"] == cfg.policy_steps
            report["observed_endpoint_decision"], report["endpoint_gates"] = endpoint_decision(report["endpoints"], report["comparisons"], regularization)
            report["decision"] = ("smoke_only" if smoke else "unresolved_density_validity_stop" if report["state"] != "completed"
                                  else report["observed_endpoint_decision"])
            if controller:
                report["final_density"] = density_panel(controller, model, data, cfg, seed+109000)
                report["final_density_certificate"] = controller.current.certificate
                report["final_autodiff_validation"] = instrument()
            report["frozen_models_unchanged"] = unchanged(initial, payload["initial"]) and unchanged(strong, payload["classifier"]) and unchanged(weak, weak_state)
            if not report["frozen_models_unchanged"] or any(p.grad is not None for m in (weak, strong) for p in m.parameters()):
                raise RuntimeError("Frozen source/classifier was changed")
            report["seconds"] = time.perf_counter()-started
            save()
            emit({"phase": "complete", "step": report["completed_steps"], "decision": report["decision"],
                  "endpoint_gates": report["endpoint_gates"], "seconds": report["seconds"]})
            return report
        except BaseException as exc:
            if "last_valid" in locals():
                torch.save(last_valid, output/"dgpo_state.pt")
                report.update(completed_steps=last_valid["step"], policy_history=last_valid["history"])
            report.update(state="error", decision="interrupted_or_error", error={"type": type(exc).__name__, "message": str(exc)})
            save()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--plain-checkpoint", required=True, type=Path)
    parser.add_argument("--comparator", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--regularization", choices=("endpoint", "velocity", "none"), default="endpoint")
    parser.add_argument("--policy-steps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.source, args.comparator, args.output, regularization=args.regularization,
        plain_checkpoint=args.plain_checkpoint, policy_steps=args.policy_steps, seed=args.seed, smoke=args.smoke)


if __name__ == "__main__":
    main()
