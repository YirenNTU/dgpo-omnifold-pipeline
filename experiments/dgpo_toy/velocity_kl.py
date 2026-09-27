"""Matched step-0 DGPO arm with legacy velocity-MSE regularization."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import time

import torch

try:
    from .conditional import Denoiser, Distribution, paired_gain, policy_panel, policy_train
    from .fixed_reward import load_source, panel_summary, unchanged, REQUIRED_GATES, WAIVED_GATES
    from .structure_metrics import structure_panel, structure_target, paired_structure_change
except ImportError:
    from conditional import Denoiser, Distribution, paired_gain, policy_panel, policy_train
    from fixed_reward import load_source, panel_summary, unchanged, REQUIRED_GATES, WAIVED_GATES
    from structure_metrics import structure_panel, structure_target, paired_structure_change


def load_comparator(directory, source, cfg, seed):
    report = json.loads((directory/"report.json").read_text())
    state = torch.load(directory/"dgpo.pt", map_location="cpu", weights_only=True)
    if (report["config"] != asdict(cfg) or state["config"] != asdict(cfg)
            or report["seed"] != seed or report["decision"] in ("running", "interrupted_or_error")
            or Path(report["source"]).resolve() != source.resolve()
            or Path(state["source"]).resolve() != source.resolve()
            or state["step"] != cfg.policy_steps or state["arm"] != "dgpo"
            or state.get("velocity_coefficient", 0.) != 0.
            or report.get("velocity_coefficient", 0.) != 0.
            or report["stream_seeds"]["monitor"] != seed+90000
            or report["stream_seeds"]["train"] != seed+3000):
        raise ValueError("Comparator must be completed no-KL with matched source/config/seed")
    history = report["policy_histories"]["dgpo"]
    if len(history) != cfg.policy_steps or history[-1]["step"] != cfg.policy_steps:
        raise ValueError("Comparator history is incomplete")
    model = Denoiser(cfg).eval().requires_grad_(False)
    model.load_state_dict(state["model"], strict=True)
    return model, report


def run(source, comparator, output, *, coefficient=1., policy_steps=10000,
        seed=17, endpoint_seed=94017):
    if not math.isfinite(coefficient) or coefficient <= 0 or policy_steps < 1:
        raise ValueError("Positive finite coefficient and positive policy_steps required")
    if output.resolve() in (source.parent.resolve(), comparator.resolve()):
        raise ValueError("Use a separate output to preserve source and comparator")
    if endpoint_seed in (seed+3000, seed+90000):
        raise ValueError("Endpoint stream must differ from train and monitor")
    cfg, initial, critic, payload, evidence = load_source(source)
    cfg = replace(cfg, policy_steps=policy_steps)
    gates = {k: evidence["validity"].get(k) is True for k in REQUIRED_GATES}
    if not all(gates.values()):
        raise ValueError(f"Unwaived source gates failed: {[k for k, v in gates.items() if not v]}")
    control, previous = load_comparator(comparator, source, cfg, seed)
    data = Distribution(cfg)
    control_monitor, _ = policy_panel(control, critic, data, cfg, seed+90000)
    if float(control_monitor.mean()) != previous["policy_histories"]["dgpo"][-1]["monitor_reward"]:
        raise ValueError("Comparator endpoint does not reproduce its monitor")
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    report = {"experiment": "conditional_velocity_mse_ablation_v1", "seed": seed,
              "source": str(source.resolve()), "comparator": str(comparator.resolve()),
              "config": asdict(cfg), "velocity_coefficient": coefficient,
              "formula": "L_DGPO + coefficient * 0.5 * mean((v_current-v_step0)^2)",
              "classifier_selected_step": payload["selected_step"], "feature_mode": payload["feature_mode"],
              "source_confirmation": evidence["confirmation"], "validity": gates,
              "waived_for_fixed_reward_question": list(WAIVED_GATES),
              "stream_seeds": {"train": seed+3000, "monitor": seed+90000, "endpoint": endpoint_seed},
              "endpoints": {}, "policy_histories": {"no_kl": previous["policy_histories"]["dgpo"]},
              "decision": "running"}

    def save():
        (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")

    with (output/"progress.jsonl").open("w") as log:
        def emit(row):
            line = json.dumps({"seed": seed, **row}, allow_nan=False)
            print(line, flush=True)
            log.write(line+"\n")
            log.flush()

        def checkpoint(step, model, optimizer, rng, history):
            if step == 1:
                prior = previous["policy_histories"]["dgpo"][0]
                matches = all(history[0][key] == value for key, value in prior.items())
                report["first_update_matches_no_kl"] = matches
                if not matches:
                    raise RuntimeError("First step should match no-KL exactly (zero penalty gradient)")
            if step % 1000 == 0 or step == cfg.policy_steps:
                torch.save({"config": asdict(cfg), "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(), "rng": rng.get_state(),
                            "history": history, "seed": seed, "step": step, "arm": "dgpo",
                            "velocity_coefficient": coefficient, "source": str(source.resolve()),
                            "monitor_seed": seed+90000}, output/"dgpo_state.pt")
                save()

        try:
            save()
            model, history = policy_train("dgpo", initial, critic, data, cfg, seed,
                                          seed+90000, emit, checkpoint,
                                          velocity_coefficient=coefficient)
            report["policy_histories"]["velocity_kl"] = history
            panels, structures = {}, {}
            for name, policy in (("initial", initial), ("no_kl", control), ("velocity_kl", model)):
                reward, joint = policy_panel(policy, critic, data, cfg, endpoint_seed)
                panels[name] = (reward, joint)
                structures[name], structural = structure_panel(policy, data, cfg, endpoint_seed)
                report["endpoints"][name] = {"final": panel_summary(reward, joint), "structure": structural}
            target = structure_target(data)
            for name in ("no_kl", "velocity_kl"):
                result = report["endpoints"][name]
                result.update(paired_gain(panels[name][0], panels["initial"][0]))
                result["joint_change"] = paired_gain(panels[name][1], panels["initial"][1])
                result["structure_change"] = paired_structure_change(
                    structures[name], structures["initial"], target, endpoint_seed+1)
                result["reward_threshold_pass"] = result["gain"] >= .1 and result["lo95"] > 0
            contrast = paired_gain(panels["velocity_kl"][0], panels["no_kl"][0])
            report["reward_contrast_velocity_minus_no_kl"] = contrast
            report["structure_contrast_velocity_minus_no_kl"] = paired_structure_change(
                structures["velocity_kl"], structures["no_kl"], target, endpoint_seed+2)
            report["decision"] = ("reward_improved_by_velocity" if contrast["lo95"] > 0 else
                                  "reward_suppressed_by_velocity" if contrast["hi95"] < 0 else "unresolved")
            report["frozen_source_unchanged"] = unchanged(initial, payload["initial"]) and unchanged(critic, payload["classifier"])
            if not report["frozen_source_unchanged"] or any(p.grad is not None for p in critic.parameters()):
                raise RuntimeError("Frozen source or classifier changed")
            torch.save({"config": asdict(cfg), "model": model.state_dict(), "source": str(source.resolve()),
                        "step": cfg.policy_steps, "arm": "dgpo", "velocity_coefficient": coefficient}, output/"dgpo.pt")
            report["seconds"] = time.perf_counter()-started
            save()
            emit({"phase": "complete", "decision": report["decision"], "seconds": report["seconds"],
                  "reward_contrast": contrast,
                  "structure_contrast": report["structure_contrast_velocity_minus_no_kl"]})
            return report
        except BaseException as exc:
            report["decision"] = "interrupted_or_error"
            report["error"] = {"type": type(exc).__name__, "message": str(exc)}
            save()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--comparator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coefficient", type=float, default=1.)
    parser.add_argument("--policy-steps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--endpoint-seed", type=int, default=94017)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.source, args.comparator, args.output, coefficient=args.coefficient,
        policy_steps=args.policy_steps, seed=args.seed, endpoint_seed=args.endpoint_seed)


if __name__ == "__main__":
    main()
