"""Three fixed-reward arms from the same raw, truth-trained diffusion."""
from __future__ import annotations
import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
import torch
from .conditional import Classifier, Distribution, classifier_metrics, policy_train, paired_gain
from .low_ess_recovery import JointFourierClassifier
from .matched_h4 import load_inputs
from .fixed_reward import unchanged
from .weak_classifier import paired_scores, signal_metrics
from .structure_metrics import structure_panel, structure_target, paired_structure_change
from .truth_pretrain import atomic_json, atomic_checkpoint

ARMS = {"strong_velocity": ("strong", 1.), "weak_velocity": ("weak", 1.),
        "strong_no_kl": ("strong", 0.)}


def load_critics(paths, cfg, dataset_path, diffusion_path, dataset):
    critics, states, evidence = {}, {}, {}
    shared = None
    for name, mode in (("strong", "joint3"), ("weak", "plain")):
        path = paths[name]
        state = torch.load(path, map_location="cpu", weights_only=True)
        report = json.loads(path.with_name("report.json").read_text())
        if (state["config"] != asdict(cfg) or report["state"] != "completed"
                or state["feature_mode"] != mode or state["weights"] != "raw"
                or state["step"] != report["best_step"] or state["epoch"] != report["best_epoch"]
                or Path(state["source"]).resolve() != diffusion_path.resolve()
                or Path(state["dataset"]).resolve() != dataset_path.resolve()):
            raise ValueError("Classifier selection/source mismatch")
        panels = torch.load(path.with_name("panels.pt"), map_location="cpu", weights_only=True)
        for split, panel in panels.items():
            if not (torch.equal(panel["context"], dataset[split]["condition"])
                    and torch.equal(panel["truth"], dataset[split]["target"])):
                raise ValueError("Classifier truth rows differ")
            if shared is not None and any(not torch.equal(v, shared[split][k]) for k,v in panel.items()):
                raise ValueError("Classifier paired panels differ")
        shared = panels
        model = JointFourierClassifier(cfg) if mode == "joint3" else Classifier(cfg, False)
        model.load_state_dict(state["classifier"], strict=True)
        model.eval().requires_grad_(False)
        actual = classifier_metrics(model, panels["validation"], Distribution(cfg))
        if abs(actual["bce"]-report["selected_metrics"]["validation"]["bce"]) > 1e-6:
            raise ValueError("Selected validation BCE cannot be reproduced")
        critics[name], states[name], evidence[name] = model, state, report["selected_metrics"]
    return critics, states, evidence


def own_result(after, before):
    scale = float(before.std(unbiased=False))
    if scale <= 0:
        raise ValueError("Constant reward cannot be standardized")
    raw = paired_gain(after, before)
    normalized = {k: v/scale for k,v in raw.items()}
    return {"raw": raw, "initial_reward_std": scale, "standardized": normalized,
            "reliable_absorption": raw["lo95"] > 0,
            "nontrivial_absorption": raw["lo95"] > 0 and normalized["gain"] >= .01}


def run(output, *, dataset_path, diffusion_path, strong_path, weak_path, steps=10000, seed=17):
    if steps < 1:
        raise ValueError("Positive update budget required")
    cfg, dataset, initial, source = load_inputs(dataset_path, diffusion_path)
    critics, states, evidence = load_critics({"strong": strong_path, "weak": weak_path},
                                            cfg, dataset_path, diffusion_path, dataset)
    cfg = replace(cfg, policy_steps=steps)
    output.mkdir(parents=True, exist_ok=False)
    data = Distribution(cfg)
    monitor_seed, endpoint_seed = seed+130000, seed+132000
    report = {"state": "running", "config": asdict(cfg), "seed": seed,
        "dataset": str(dataset_path.resolve()), "source": str(diffusion_path.resolve()),
        "classifiers": {k: str(p.resolve()) for k,p in (("strong",strong_path),("weak",weak_path))},
        "classifier_metrics": evidence, "monitor_seed": monitor_seed, "endpoint_seed": endpoint_seed,
        "arms": {}, "caveat": "Single seed; architecture and reward scale confound ESS. MSE is a KL surrogate, not exact KL."}
    atomic_json(output/"report.json", report)
    base = paired_scores(initial, critics["weak"], critics["strong"], data, cfg, endpoint_seed)
    base_structure, base_summary = structure_panel(initial, data, cfg, endpoint_seed)
    report["initial"] = {"signal": signal_metrics(base), "structure": base_summary}
    endpoints = {}
    try:
        for arm, (judge, coefficient) in ARMS.items():
            directory = output/arm
            directory.mkdir()
            with (directory/"progress.jsonl").open("w") as stream:
                def emit(row):
                    row = {**row, "arm": arm}
                    stream.write(json.dumps(row, allow_nan=False)+"\n")
                    stream.flush()
                    if row["step"] == 1 or row["step"] % 1000 == 0 or row["step"] == steps:
                        print(json.dumps(row, allow_nan=False), flush=True)
                def checkpoint(step, model, optimizer, rng, history):
                    if step == 1 or step % 250 == 0 or step == steps:
                        scores = paired_scores(model, critics["weak"], critics["strong"], data, cfg, monitor_seed)
                        _, structure = structure_panel(model, data, cfg, monitor_seed)
                        emit({"phase": "common_diagnostics", "step": step,
                              "signal": signal_metrics(scores), "structure": structure})
                    if step == 1 or step % 1000 == 0 or step == steps:
                        atomic_checkpoint(directory/"state.pt", {"model": model.state_dict(),
                            "optimizer": optimizer.state_dict(), "rng": rng.get_state(), "history": history,
                            "step": step, "config": asdict(cfg), "seed": seed, "arm": arm,
                            "source": report["source"], "classifiers": report["classifiers"],
                            "monitor_seed": monitor_seed, "velocity_coefficient": coefficient,
                            "endpoint_controller": None})
                model, history = policy_train("dgpo", initial, critics[judge], data, cfg, seed,
                    monitor_seed, emit, checkpoint, velocity_coefficient=coefficient)
            scores = paired_scores(model, critics["weak"], critics["strong"], data, cfg, endpoint_seed)
            structure, summary = structure_panel(model, data, cfg, endpoint_seed)
            endpoints[arm] = scores
            report["arms"][arm] = {"steps": len(history), "judge": judge, "velocity_coefficient": coefficient,
                "own_reward": own_result(scores[judge], base[judge]), "signal": signal_metrics(scores),
                "common_reward_gains": {k: paired_gain(scores[k], base[k]) for k in base},
                "structure": summary, "structure_change": paired_structure_change(
                    structure, base_structure, structure_target(data), endpoint_seed+1)}
            atomic_json(directory/"report.json", report["arms"][arm])
            atomic_json(output/"report.json", report)
        report["contrasts"] = {
            "strong_no_kl_minus_velocity": paired_gain(endpoints["strong_no_kl"]["strong"], endpoints["strong_velocity"]["strong"]),
            "weak_minus_strong_standardized_gain": paired_gain(
                (endpoints["weak_velocity"]["weak"]-base["weak"])/base["weak"].std(unbiased=False),
                (endpoints["strong_velocity"]["strong"]-base["strong"])/base["strong"].std(unbiased=False))}
        report["sources_unchanged"] = unchanged(initial, source["model"]) and all(
            unchanged(critics[k], states[k]["classifier"]) for k in critics)
        if not report["sources_unchanged"]:
            raise RuntimeError("Frozen source mutated")
        atomic_checkpoint(output/"endpoint_scores.pt", {"initial": base, **endpoints})
        report["state"] = "completed"
    except BaseException as exc:
        report.update(state="failed_or_interrupted", error=repr(exc))
        raise
    finally:
        atomic_json(output/"report.json", report)
    print(json.dumps({"state": report["state"], "contrasts": report["contrasts"]}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "dataset-path", "diffusion-path", "strong-path", "weak-path"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(**vars(args))


if __name__ == "__main__":
    main()
