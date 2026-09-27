"""Direct numerical endpoint log ratio BEFORE the DGPO advantage. No KL critic."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import torch

try:
    from .conditional import Distribution, Denoiser, generator, ddim, policy_train, policy_panel, paired_gain
    from .fixed_reward import load_source, unchanged, panel_summary, REQUIRED_GATES, WAIVED_GATES
    from .velocity_kl import load_comparator
    from .structure_metrics import structure_panel, structure_target, paired_structure_change
    from .direct_density import DDIMDensity, DensitySettings, DensityValidityError, EndpointRatio
except ImportError:
    from conditional import Distribution, Denoiser, generator, ddim, policy_train, policy_panel, paired_gain
    from fixed_reward import load_source, unchanged, panel_summary, REQUIRED_GATES, WAIVED_GATES
    from velocity_kl import load_comparator
    from structure_metrics import structure_panel, structure_target, paired_structure_change
    from direct_density import DDIMDensity, DensitySettings, DensityValidityError, EndpointRatio


def decision(result, initial_structure):
    s = result["structure"]
    transfer = result["gain"] >= .1 and result["lo95"] > 0
    joint = result["structure_change"]["hi95"] < 0 and s["fourier_moment_rmse"] <= .9*initial_structure["fourier_moment_rmse"]
    lower = s["marginal_mean_absmax"] <= .10 and s["marginal_variance_error_absmax"] <= .15 and s["pair_covariance_absmax"] <= .08
    return ("joint_alignment_progress" if transfer and joint and lower else
            "reward_transfer_without_alignment" if transfer else "no_decisive_alignment"), {
                "reward_transfer": transfer, "high_order_improvement": joint, "lower_order_preserved": lower}


@torch.no_grad()
def density_panel(controller, model, data, cfg, seed, count=None):
    rng = generator(seed)
    count = cfg.eval_events if count is None else count
    c = data.contexts(count, rng)
    z = torch.randn(count, cfg.candidates, cfg.dimensions, generator=rng)
    scores = []
    for cc, zz in zip(c.split(64), z.split(64)):
        y = ddim(model, cc[:, None], zz, cfg.ddim_steps)
        scores.append(controller(y, cc[:, None].expand(-1, cfg.candidates, -1)).double())
    scores = torch.cat(scores)
    groups = scores.mean(-1)
    se = float(groups.std(unbiased=True)/len(groups)**.5) if len(groups) > 1 else 0.
    return {"kl_mean": float(groups.mean()), "kl_context_se": se,
            "kl_lo95": float(groups.mean())-1.96*se, "kl_hi95": float(groups.mean())+1.96*se,
            "log_ratio_std": float(scores.std(unbiased=False)), "contexts": count,
            "candidates_per_context": cfg.candidates}


def run(source, comparator, output, *, velocity_comparator=None, policy_steps=10000,
        seed=17, endpoint_seed=96017, settings=DensitySettings(), resume_from=None,
        comparator_steps=None):
    protected = [source.parent, comparator]+([velocity_comparator] if velocity_comparator else [])
    if resume_from is not None:
        protected.append(resume_from)
    if any(output.resolve() == p.resolve() for p in protected):
        raise ValueError("Use a separate output to preserve source and controls")
    if policy_steps < 1 or endpoint_seed in (seed+3000, seed+90000):
        raise ValueError("Positive policy budget and independent endpoint stream required")
    cfg, initial, critic, payload, evidence = load_source(source)
    cfg = replace(cfg, policy_steps=policy_steps)
    comparator_steps = policy_steps if comparator_steps is None else comparator_steps
    if comparator_steps < 1:
        raise ValueError("Comparator step budget must be positive")
    comparator_cfg = replace(cfg, policy_steps=comparator_steps)
    gates = {k: evidence["validity"].get(k) is True for k in REQUIRED_GATES}
    if not all(gates.values()):
        raise ValueError("Unwaived source validity gate failed")
    control, previous = load_comparator(comparator, source, comparator_cfg, seed)
    data = Distribution(cfg)
    if float(policy_panel(control, critic, data, cfg, seed+90000)[0].mean()) != previous["policy_histories"]["dgpo"][-1]["monitor_reward"]:
        raise ValueError("No-KL comparator monitor mismatch")
    policies = {"initial": initial, "no_kl": control}
    policy_budgets = {"initial": 0, "no_kl": comparator_steps}
    histories = {"no_kl": previous["policy_histories"]["dgpo"]}
    if velocity_comparator is not None:
        vp = json.loads((velocity_comparator/"report.json").read_text())
        state = torch.load(velocity_comparator/"dgpo.pt", map_location="cpu", weights_only=True)
        if (state["config"] != asdict(comparator_cfg) or state["step"] != comparator_steps or vp["seed"] != seed
                or Path(state["source"]).resolve() != source.resolve()
                or vp["velocity_coefficient"] != 1. or state["velocity_coefficient"] != 1.):
            raise ValueError("Velocity comparator must match source/config/seed and coefficient1")
        model = Denoiser(cfg).eval().requires_grad_(False)
        model.load_state_dict(state["model"])
        policies["velocity_kl"] = model
        policy_budgets["velocity_kl"] = comparator_steps
        histories["velocity_kl"] = vp["policy_histories"]["velocity_kl"]
    controller = EndpointRatio(settings)
    resume_state = None
    if resume_from is not None:
        resume_state = torch.load(resume_from/"dgpo_state.pt", map_location="cpu", weights_only=True)
        if ({**resume_state["config"], "policy_steps": policy_steps} != asdict(cfg)
                or Path(resume_state["source"]).resolve() != source.resolve()
                or resume_state["seed"] != seed or resume_state["monitor_seed"] != seed+90000
                or resume_state["endpoint_controller"]["policy_step"] != resume_state["step"]
                or not 0 < resume_state["step"] < policy_steps
                or len(resume_state["history"]) != resume_state["step"]
                or any(row["step"] != i+1 for i, row in enumerate(resume_state["history"]))
                or not resume_state["optimizer"]["state"]
                or any(float(s["step"]) != resume_state["step"] for s in resume_state["optimizer"]["state"].values())):
            raise ValueError("Resume provenance/clock mismatch")
        controller.load_state_dict(resume_state["endpoint_controller"])
        boundary = copy.deepcopy(initial)
        boundary.load_state_dict(resume_state["model"])
        policies["resume_start"] = boundary
        policy_budgets["resume_start"] = resume_state["step"]
        boundary_reward, boundary_joint = policy_panel(boundary, critic, data, cfg, seed+90000)
        expected = resume_state["history"][-1]
        if (float(boundary_reward.mean()) != expected["monitor_reward"]
                or float(boundary_reward.std(unbiased=False)) != expected["monitor_reward_std"]
                or float(boundary_joint.mean()) != expected["monitor_joint_signal"]):
            raise ValueError("Resume boundary monitor mismatch")
    start_step = resume_state["step"] if resume_state is not None else 0
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report = {"experiment": "direct_endpoint_net_reward_dgpo_v1", "source": str(source.resolve()),
              "comparator": str(comparator.resolve()), "velocity_comparator": str(velocity_comparator) if velocity_comparator else None,
              "config": asdict(cfg), "density_settings": asdict(settings), "seed": seed,
              "stream_seeds": {"train": seed+3000, "monitor": seed+90000, "endpoint": endpoint_seed},
              "source_validity": evidence["validity"], "validity": gates,
              "waived_for_fixed_reward_question": list(WAIVED_GATES),
              "classifier_selected_step": payload["selected_step"], "feature_mode": payload["feature_mode"],
              "formula": "LOO(frozen_H4_logit - coefficient*(log_q_current-log_q_step0)) -> DGPO gate/loss",
              "density_definition": "Continuous DDIM map in float64 at actual rollout points; numerical checks, not interval proof",
              "policy_histories": histories, "endpoints": {}, "decision": "running", "completed_steps": start_step,
              "start_step": start_step, "requested_new_updates": policy_steps-start_step,
              "new_updates_completed": 0, "comparator_steps": comparator_steps}
    if resume_state is not None:
        report["resume_from"] = str(resume_from.resolve())
        report["resume_verification"] = {"monitor_boundary_exact": True, "history_prefix_exact": True,
                                         "optimizer_clock": start_step, "density_clock": start_step}
        report["policy_histories"]["net_reward_kl"] = copy.deepcopy(resume_state["history"])

    def save():
        (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")

    def snapshot(step, model, optimizer, rng, history):
        return {"config": asdict(cfg), "model": copy.deepcopy(model.state_dict()),
                "optimizer": copy.deepcopy(optimizer.state_dict()), "rng": rng.get_state().clone(),
                "history": list(history), "seed": seed, "step": step, "arm": "dgpo",
                "source": str(source.resolve()), "monitor_seed": seed+90000, "velocity_coefficient": 0.,
                "endpoint_controller": controller.state_dict()}

    last_valid = (copy.deepcopy(resume_state) if resume_state is not None else snapshot(
        0, initial, torch.optim.AdamW(initial.parameters(), lr=cfg.policy_lr, weight_decay=cfg.weight_decay),
        generator(seed+3000), []))
    with (output/"progress.jsonl").open("w") as log:
        def emit(row):
            line = json.dumps({"seed": seed, **row}, allow_nan=False)
            print(line, flush=True)
            log.write(line+"\n")
            log.flush()

        def instrument():
            rng = generator(180017)
            cc = data.contexts(8, rng)
            zz = torch.randn(8, cfg.dimensions, generator=rng)
            return controller.current.validate_autodiff(cc, zz)

        def checkpoint(step, model, optimizer, rng, history):
            nonlocal last_valid
            # Certify post-update too, including the final update. No projection.
            try:
                controller.refresh(model, initial, cfg.ddim_steps)
                if step == 1 or step % settings.structure_every == 0 or step == cfg.policy_steps:
                    history[-1].update({"density/autodiff_"+k: v for k, v in instrument().items()})
            except DensityValidityError:
                torch.save({"model": model.state_dict(), "step": step, "status": "uncertified_attempt"}, output/"uncertified_attempt.pt")
                raise
            if step == 1:
                first = previous["policy_histories"]["dgpo"][0]
                report["first_update_matches_no_kl"] = all(history[0][k] == v for k, v in first.items())
                if not report["first_update_matches_no_kl"]:
                    raise RuntimeError("Zero initial KL must preserve first no-KL update")
            if step % settings.structure_every == 0 or step == cfg.policy_steps:
                _, values = structure_panel(model, data, cfg, seed+90000)
                values = {"monitor_structure_"+k: v for k, v in values.items() if k not in ("moment_mean", "truth_moment")}
                density = density_panel(controller, model, data, cfg, seed+190000, count=min(512, cfg.eval_events))
                values.update({"monitor_density_"+k: v for k, v in density.items()})
                history[-1].update(values)
                emit({"phase": "structure_density_monitor", "step": step, **values})
            last_valid = snapshot(step, model, optimizer, rng, history)
            report["completed_steps"] = step
            report["new_updates_completed"] = step-start_step
            report["policy_histories"]["net_reward_kl"] = history
            if resume_state is not None and step == start_step+1:
                if history[:start_step] != resume_state["history"]:
                    raise RuntimeError("Resume changed the historical trajectory")
                if any(float(s["step"]) != step for s in optimizer.state_dict()["state"].values()):
                    raise RuntimeError("Resume reset the optimizer clock")
                emit({"phase": "resume_first_update_verified", "step": step, "new_updates": 1})
            if step % 1000 == 0 or step == cfg.policy_steps or (resume_state is not None and step == start_step+1):
                torch.save(last_valid, output/"dgpo_state.pt")
                save()

        try:
            save()
            controller.refresh(initial, initial, cfg.ddim_steps)
            report["initial_density_validation"] = {"certificate": controller.current.certificate, "autodiff": instrument()}
            if resume_state is not None:
                controller.refresh(policies["resume_start"], initial, cfg.ddim_steps)
                report["resume_density_validation"] = {"certificate": controller.current.certificate, "autodiff": instrument()}
                emit({"phase": "resume_verified", "step": start_step, **report["resume_verification"]})
            try:
                model, history = policy_train("dgpo", initial, critic, data, cfg, seed, seed+90000,
                    emit, checkpoint, resume_state, endpoint_controller=controller)
                report["state"] = "completed"
            except DensityValidityError as exc:
                report["state"] = "stopped_density_validity"
                report["stop"] = {"attempted_step": last_valid["step"]+1, "message": str(exc),
                                  "interpretation": "Sufficient validity condition failed; not proof of noninvertibility"}
                model = copy.deepcopy(initial)
                model.load_state_dict(last_valid["model"])
                history = last_valid["history"]
                controller.load_state_dict(last_valid["endpoint_controller"])
                controller.refresh(model, initial, cfg.ddim_steps)
                emit({"phase": "validity_stop", **report["stop"], "last_valid_step": last_valid["step"]})
            torch.save(last_valid, output/"dgpo_state.pt")
            policies["net_reward_kl"] = model
            policy_budgets["net_reward_kl"] = last_valid["step"]
            report["completed_steps"] = last_valid["step"]
            report["new_updates_completed"] = last_valid["step"]-start_step
            report["policy_histories"]["net_reward_kl"] = history
            report["comparison_budgets_matched"] = last_valid["step"] == comparator_steps
            panels, structures = {}, {}
            for name, policy in policies.items():
                reward, joint = policy_panel(policy, critic, data, cfg, endpoint_seed)
                panels[name] = (reward, joint)
                structures[name], values = structure_panel(policy, data, cfg, endpoint_seed)
                steps = policy_budgets[name]
                report["endpoints"][name] = {"step": steps, "final": panel_summary(reward, joint), "structure": values}
            for name in policies.keys()-{"initial"}:
                result = report["endpoints"][name]
                result.update(paired_gain(panels[name][0], panels["initial"][0]))
                result["joint_change"] = paired_gain(panels[name][1], panels["initial"][1])
                result["structure_change"] = paired_structure_change(structures[name], structures["initial"], structure_target(data), endpoint_seed+1)
            result = report["endpoints"]["net_reward_kl"]
            if resume_state is not None:
                result["gain_since_resume"] = paired_gain(panels["net_reward_kl"][0], panels["resume_start"][0])
                result["joint_change_since_resume"] = paired_gain(panels["net_reward_kl"][1], panels["resume_start"][1])
                result["structure_change_since_resume"] = paired_structure_change(
                    structures["net_reward_kl"], structures["resume_start"], structure_target(data), endpoint_seed+3)
                report["resume_verification"]["history_prefix_exact"] = history[:start_step] == resume_state["history"]
                if not report["resume_verification"]["history_prefix_exact"]:
                    raise RuntimeError("Resume changed the historical trajectory")
            result["density"] = density_panel(controller, model, data, cfg, endpoint_seed)
            result["net_reward_mean"] = result["final"]["reward_mean"]-settings.coefficient*result["density"]["kl_mean"]
            candidate, report["endpoint_gates"] = decision(result, report["endpoints"]["initial"]["structure"])
            if resume_state is not None:
                report["extension_gates"] = {
                    "reward_increased_since_resume": result["gain_since_resume"]["lo95"] > 0,
                    "high_order_improved_since_resume": result["structure_change_since_resume"]["hi95"] < 0,
                    "lower_order_preserved": report["endpoint_gates"]["lower_order_preserved"]}
                report["extension_decision"] = (
                    "unresolved_density_validity_stop" if report["state"] != "completed" else
                    "supports_accumulation" if all(report["extension_gates"].values()) else
                    "unresolved_at_extended_budget")
            report["observed_endpoint_classification"] = candidate
            report["decision"] = candidate if report["state"] == "completed" else "unresolved_density_validity_stop"
            report["final_density_certificate"] = controller.current.certificate
            report["final_autodiff_validation"] = instrument()
            report["comparisons"] = {name: {
                "matched_update_budget": report["comparison_budgets_matched"],
                "raw_reward_net_arm_minus_control": paired_gain(panels["net_reward_kl"][0], panels[name][0]),
                "structure_net_arm_minus_control": paired_structure_change(structures["net_reward_kl"], structures[name], structure_target(data), endpoint_seed+2)
            } for name in ("no_kl", "velocity_kl") if name in policies}
            torch.save({"config": asdict(cfg), "model": model.state_dict(), "source": str(source.resolve()),
                        "step": last_valid["step"], "arm": "net_reward_kl"}, output/"dgpo.pt")
            report["frozen_source_unchanged"] = unchanged(initial, payload["initial"]) and unchanged(critic, payload["classifier"])
            if not report["frozen_source_unchanged"] or any(p.grad is not None for p in critic.parameters()):
                raise RuntimeError("Fixed H4/source was modified")
            report["seconds"] = time.perf_counter()-started
            save()
            emit({"phase": "complete", "decision": report["decision"], "seconds": report["seconds"],
                  "completed_steps": last_valid["step"], "endpoint_gates": report["endpoint_gates"],
                  "reward_gain": result["gain"], "structure_change": result["structure_change"]})
            return report
        except BaseException as exc:
            torch.save(last_valid, output/"dgpo_state.pt")
            report["decision"] = "interrupted_or_error"
            report["error"] = {"type": type(exc).__name__, "message": str(exc)}
            report["completed_steps"] = last_valid["step"]
            report["new_updates_completed"] = last_valid["step"]-start_step
            report["policy_histories"]["net_reward_kl"] = last_valid["history"]
            save()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--comparator", type=Path, required=True)
    parser.add_argument("--velocity-comparator", type=Path)
    parser.add_argument("--comparator-steps", type=int,
                        help="Saved control budget (defaults to --policy-steps); differing budgets are labelled unmatched")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-steps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--endpoint-seed", type=int, default=96017)
    parser.add_argument("--coefficient", type=float, default=1.)
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.source, args.comparator, args.output, velocity_comparator=args.velocity_comparator,
        policy_steps=args.policy_steps, seed=args.seed, endpoint_seed=args.endpoint_seed,
        settings=DensitySettings(coefficient=args.coefficient), resume_from=args.resume_from,
        comparator_steps=args.comparator_steps)


if __name__ == "__main__":
    main()
