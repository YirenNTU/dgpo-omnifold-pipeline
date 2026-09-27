"""Train policy against the saved low-ESS classifier without fitting it again."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import torch

try:
    from .conditional import (Config, Distribution, Denoiser, Classifier,
        policy_panel, policy_train, paired_gain, weight_health, decide)
    from .low_ess_recovery import JointFourierClassifier
except ImportError:
    from conditional import (Config, Distribution, Denoiser, Classifier,
        policy_panel, policy_train, paired_gain, weight_health, decide)
    from low_ess_recovery import JointFourierClassifier


WAIVED_GATES = ("ratio_normalization", "truth_logit_rms")
REQUIRED_GATES = ("marginals", "variance", "pair_covariance", "conditional_mean",
    "classifier_bce", "classifier_auc", "low_ess", "within_context_signal",
    "missing_joint_structure", "inverse_converged", "inverse_start_agreement",
    "jacobian_condition", "reference_bce_agreement")


def load_source(source):
    payload = torch.load(source, map_location="cpu", weights_only=True)
    evidence = json.loads(source.with_name("report.json").read_text())
    if payload["config"] != evidence["config"]:
        raise ValueError("Source checkpoint and report configurations disagree")
    if (payload["selected_step"] != evidence["fit"]["selected_step"]
            or payload["feature_mode"] != evidence["feature_mode"]):
        raise ValueError("Source checkpoint and report classifier selections disagree")
    cfg = Config(**payload["config"])
    initial = Denoiser(cfg)
    initial.load_state_dict(payload["initial"], strict=True)
    if payload["feature_mode"] == "joint3":
        critic = JointFourierClassifier(cfg)
    elif payload["feature_mode"] == "coordinate":
        critic = Classifier(cfg, True)
    else:
        raise ValueError("Unsupported saved classifier feature mode")
    critic.load_state_dict(payload["classifier"], strict=True)
    for model in (initial, critic):
        model.eval().requires_grad_(False)
        if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
            raise ValueError("Nonfinite source model")
    return cfg, initial, critic, payload, evidence


def unchanged(model, state):
    return all(torch.equal(value, state[key]) for key, value in model.state_dict().items())


def panel_summary(reward, joint):
    if not torch.isfinite(reward).all() or not torch.isfinite(joint).all():
        raise FloatingPointError("Nonfinite policy evaluation")
    return {"reward_mean": float(reward.mean()),
            "reward_std": float(reward.std(unbiased=False)),
            "joint_signal": float(joint.mean()), **weight_health(reward)}


def load_resume_state(directory, arm, source, cfg, seed, previous):
    """Validate full state, not a weights-only restart or recentered reference."""
    state = torch.load(directory/(arm+"_state.pt"), map_location="cpu", weights_only=True)
    expected = {**state["config"], "policy_steps": cfg.policy_steps}
    if (expected != asdict(cfg) or state["seed"] != seed or state["arm"] != arm
            or Path(state["source"]).resolve() != source.resolve()
            or state["monitor_seed"] != seed+90000):
        raise ValueError("Resume must preserve source, arm, seed and training settings")
    step = state["step"]
    if (step != previous["config"]["policy_steps"] or not 0 < step < cfg.policy_steps
            or len(state["history"]) != step
            or state["history"] != previous["policy_histories"][arm]):
        raise ValueError("Resume checkpoint/history disagrees with completed run")
    if not state["optimizer"]["state"] or any(
            float(value["step"]) != step for value in state["optimizer"]["state"].values()):
        raise ValueError("Resume AdamW clock disagrees with policy step")
    endpoint = torch.load(directory/(arm+".pt"), map_location="cpu", weights_only=True)
    if not all(torch.equal(value, endpoint["model"][key]) for key, value in state["model"].items()):
        raise ValueError("Resume state and endpoint weights disagree")
    return state


def run(source: Path, output: Path, seed=17, smoke=False, policy_steps=None,
        replay_from: Path | None = None, endpoint_seed=None, resume_from: Path | None = None):
    if output.resolve() == source.resolve().parent:
        raise ValueError("Use a separate policy output to preserve the classifier source")
    if replay_from is not None and output.resolve() == replay_from.resolve():
        raise ValueError("Use a separate output to preserve the shorter policy run")
    if resume_from is not None and output.resolve() == resume_from.resolve():
        raise ValueError("Use a separate output to preserve the resume source")
    if resume_from is not None and (replay_from is not None or smoke):
        raise ValueError("Resume cannot be combined with replay or smoke")
    cfg, initial, critic, payload, evidence = load_source(source)
    if policy_steps is not None:
        if policy_steps < 1:
            raise ValueError("policy_steps must be positive")
        cfg = replace(cfg, policy_steps=policy_steps)
    if smoke:
        cfg = replace(cfg, policy_steps=2, eval_events=32, batch=4,
                      candidates=4, timesteps=2, eval_every=1)
    final_seed = seed+91000 if endpoint_seed is None else endpoint_seed
    if final_seed in (seed+3000, seed+90000):
        raise ValueError("Endpoint stream must differ from train and monitor")
    replay = None
    if replay_from is not None:
        replay = json.loads((replay_from/"report.json").read_text())
        expected_cfg = {**replay["config"], "policy_steps": cfg.policy_steps}
        if (expected_cfg != asdict(cfg) or replay["seed"] != seed
                or Path(replay["source"]).resolve() != source.resolve()
                or replay["stream_seeds"]["monitor"] != seed+90000
                or replay["stream_seeds"]["train"] != seed+3000
                or replay["config"]["policy_steps"] >= cfg.policy_steps):
            raise ValueError("Replay must change only policy_steps and endpoint evaluation stream")
    resume_states = {}
    if resume_from is not None:
        previous = json.loads((resume_from/"report.json").read_text())
        if (Path(previous["source"]).resolve() != source.resolve()
                or previous["classifier_selected_step"] != payload["selected_step"]
                or previous["feature_mode"] != payload["feature_mode"]):
            raise ValueError("Resume classifier provenance disagrees with source")
        for arm in ("dgpo", "pathwise"):
            resume_states[arm] = load_resume_state(resume_from, arm, source, cfg, seed, previous)
    gates = {name: evidence["validity"].get(name) is True for name in REQUIRED_GATES}
    if not all(gates.values()) and not smoke:
        raise ValueError(f"Unwaived setup checks failed: {[k for k, v in gates.items() if not v]}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report = {"experiment": "fixed_approximate_reward_transfer_v1", "seed": seed,
              "source": str(source.resolve()), "config": asdict(cfg),
              "classifier_selected_step": payload["selected_step"],
              "feature_mode": payload["feature_mode"],
              "source_confirmation": evidence["confirmation"],
              "source_validity": evidence["validity"],
              "waived_for_fixed_reward_question": list(WAIVED_GATES),
              "validity": gates, "stream_seeds": {"train": seed+3000,
                  "monitor": seed+90000, "endpoint": final_seed},
              "replay_from": str(replay_from.resolve()) if replay_from else None,
              "replay_verification": {},
              "resume_from": str(resume_from.resolve()) if resume_from else None,
              "resume_verification": {}, "resume_baselines": {},
              "endpoints": {}, "policy_histories": {}, "decision": "running"}

    def save():
        (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")

    with (output/"progress.jsonl").open("w") as log:
        def emit(row):
            # allow_nan=False makes even nonfinite telemetry stop immediately.
            line = json.dumps({"seed": seed, **row}, allow_nan=False)
            print(line, flush=True)
            log.write(line+"\n")
            log.flush()
        try:
            save()
            data = Distribution(cfg)
            base_reward, base_joint = policy_panel(initial, critic, data, cfg, final_seed)
            report["baseline_endpoint"] = panel_summary(base_reward, base_joint)
            emit({"phase": "baseline", **report["baseline_endpoint"]})
            save()
            for arm in ("dgpo", "pathwise"):
                arm_start = time.perf_counter()
                resume_state = resume_states.get(arm)
                resume_reward = resume_joint = None
                if resume_state is not None:
                    restored = Denoiser(cfg).eval().requires_grad_(False)
                    restored.load_state_dict(resume_state["model"], strict=True)
                    # Recheck the saved monitor boundary before applying any update.
                    monitor_reward, _ = policy_panel(restored, critic, data, cfg, seed+90000)
                    last = resume_state["history"][-1]
                    if float(monitor_reward.mean()) != last.get("monitor_reward"):
                        raise ValueError("Resumed model/critic monitor boundary does not match")
                    resume_reward, resume_joint = policy_panel(restored, critic, data, cfg, final_seed)
                    report["resume_baselines"][arm] = panel_summary(resume_reward, resume_joint)
                    report["resume_verification"][arm] = {
                        "step": resume_state["step"], "full_state_validated": True,
                        "monitor_boundary_exact": True, "optimizer_reset": False,
                        "new_optimizer_updates": cfg.policy_steps-resume_state["step"]}
                    emit({"phase": "resume", "arm": arm, **report["resume_verification"][arm]})
                    save()
                def checkpoint(step, model, optimizer, rng, history):
                    replay_step = replay["config"]["policy_steps"] if replay else None
                    if step == replay_step:
                        old = torch.load(replay_from/(arm+".pt"), map_location="cpu", weights_only=True)
                        if old["step"] != step or old["arm"] != arm:
                            raise ValueError("Replay checkpoint step/arm mismatch")
                        exact_weights = unchanged(model, old["model"])
                        exact_history = history == replay["policy_histories"][arm]
                        report["replay_verification"][arm] = {
                            "step": step, "exact_weights": exact_weights,
                            "exact_history": exact_history,
                            "optimizer_reset_at_boundary": False}
                        save()
                        if not exact_weights or not exact_history:
                            raise RuntimeError("Short-run prefix did not replay exactly; extension invalid")
                        emit({"phase": "replay_verified", "arm": arm, "step": step})
                    if step == cfg.policy_steps or step == replay_step or step % 1000 == 0:
                        # Full state for future continuation; current run never reloads it.
                        torch.save({"config": asdict(cfg), "model": model.state_dict(),
                                    "optimizer": optimizer.state_dict(), "rng": rng.get_state(),
                                    "history": history, "seed": seed, "step": step, "arm": arm,
                                    "source": str(source.resolve()), "monitor_seed": seed+90000},
                                   output/(arm+"_state.pt"))
                        save()
                model, history = policy_train(arm, initial, critic, data, cfg,
                                               seed, seed+90000, emit, checkpoint, resume_state)
                reward, joint = policy_panel(model, critic, data, cfg, final_seed)
                result = {**paired_gain(reward, base_reward),
                          "final": panel_summary(reward, joint),
                          "joint_change": paired_gain(joint, base_joint),
                          "seconds": time.perf_counter()-arm_start}
                if resume_reward is not None:
                    result["gain_since_resume"] = paired_gain(reward, resume_reward)
                    result["joint_change_since_resume"] = paired_gain(joint, resume_joint)
                report["endpoints"][arm] = result
                report["policy_histories"][arm] = history
                if not unchanged(initial, payload["initial"]) or not unchanged(critic, payload["classifier"]):
                    raise RuntimeError("Frozen classifier or initial generator changed")
                if any(p.grad is not None for p in critic.parameters()):
                    raise RuntimeError("Frozen classifier accumulated parameter gradients")
                torch.save({"config": asdict(cfg), "model": model.state_dict(),
                            "source": str(source.resolve()), "step": cfg.policy_steps,
                            "arm": arm}, output/(arm+".pt"))
                emit({"phase": "endpoint", "arm": arm, **result})
                save()
            report["frozen_source_unchanged"] = True
            report["decision"] = decide(gates, report["endpoints"], smoke)
            report["seconds"] = time.perf_counter()-started
            save()
            emit({"phase": "complete", "decision": report["decision"],
                  "seconds": report["seconds"]})
            return report
        except BaseException as exc:
            report["decision"] = "interrupted_or_error"
            report["error"] = {"type": type(exc).__name__, "message": str(exc)}
            save()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--policy-steps", type=int)
    parser.add_argument("--replay-from", type=Path,
                        help="Verify exact shorter-run prefix before extending without optimizer reset")
    parser.add_argument("--resume-from", type=Path,
                        help="Continue a completed full-state run with model/AdamW/RNG restored")
    parser.add_argument("--endpoint-seed", type=int,
                        help="Independent final evaluation stream; never used for training")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.source, args.output, args.seed, args.smoke, args.policy_steps,
        args.replay_from, args.endpoint_seed, args.resume_from)


if __name__ == "__main__":
    main()
