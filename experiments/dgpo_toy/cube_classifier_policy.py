"""Frozen learned Fourier reward versus saved oracle control on the hard cube."""
import argparse
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import torch
from .conditional import (Config, Denoiser, Classifier, generator, ddim, make_panel,
    fit_classifier, classifier_metrics, weight_health, paired_gain, policy_train)
from .parity_cube import CubeDistribution, ModeReward, cube_metrics
from .cube_oracle_policy import paired_bins
from .coverage_budget import flatten_metrics
from .truth_pretrain import atomic_json, atomic_checkpoint


class CubeFourierClassifier(Classifier):
    """Raw coordinates plus coordinate-wise Fourier features, no parity oracle."""
    def features(self, y, c, data):
        parts = [c, y]
        if self.fourier:
            # Fixed coordinate basis; joint interactions must be learned by the MLP.
            for k in range(1, self.harmonics + 1):
                angle = k * math.pi / 2 * y
                parts.extend([angle.sin(), angle.cos()])
        return torch.cat(parts, -1)


def ordering_diagnostics(learned, oracle):
    """Compare only pairs with a strict oracle preference (oracle has mode ties)."""
    a = learned[..., :, None] - learned[..., None, :]
    b = oracle[..., :, None] - oracle[..., None, :]
    valid = b.abs() > 1e-7
    ac = learned - learned.mean(-1, keepdim=True)
    bc = oracle - oracle.mean(-1, keepdim=True)
    eligible = (ac.norm(dim=-1) > 1e-8) & (bc.norm(dim=-1) > 1e-8)
    cosine = (ac * bc).sum(-1) / (ac.norm(dim=-1) * bc.norm(dim=-1)).clamp_min(1e-12)
    return {"comparable_pairs": int(valid.sum()),
        "strict_pair_agreement": float((a[valid] * b[valid] > 0).float().mean()) if valid.any() else None,
        "eligible_context_fraction": float(eligible.float().mean()),
        "centered_cosine": float(cosine[eligible].mean()) if eligible.any() else None,
        "learned_weight": weight_health(learned), "oracle_weight": weight_health(oracle)}


@torch.no_grad()
def evaluate(model, critic, data, cfg, seed):
    c = ((torch.arange(cfg.eval_events) + .5) / cfg.eval_events * 2 - 1)[:, None]
    noise = torch.randn(cfg.eval_events, cfg.candidates, 3, generator=generator(seed))
    y = torch.cat([ddim(model, cc[:, None], zz, cfg.ddim_steps)
        for cc, zz in zip(c.split(128), noise.split(128))])
    ctx = c[:, None].expand(-1, cfg.candidates, -1)
    learned = critic(y, ctx, data)
    oracle = ModeReward(data.mass)(y, ctx, data)
    hits = data.joint_signal(y, ctx)
    stats = cube_metrics(y.flatten(0, 1), ctx.flatten(0, 1), data)
    stats.update(learned_reward=float(learned.mean()), oracle_reward=float(oracle.mean()),
        ordering=ordering_diagnostics(learned, oracle))
    stats["mean_target_tv"] = sum(v["target_tv"] for v in stats["conditions"].values()) / 8
    return learned, oracle, hits, stats


def run(source, control, output, wandb_mode="offline"):
    saved = torch.load(source / "best_pretrain.pt", map_location="cpu", weights_only=True)
    metadata = json.loads((source / "report.json").read_text())
    if metadata.get("condition_frequency") is not None:
        raise ValueError("This classifier protocol only supports the original linear-condition cube")
    control_report = json.loads((control / "report.json").read_text())
    if control_report["state"] != "completed" or control_report["source"] != str((source / "best_pretrain.pt").resolve()):
        raise ValueError("Oracle control must be completed from the identical source")
    cfg = replace(Config(**saved["config"]), policy_steps=300, eval_every=50,
        classifier_steps=5000, classifier_hidden=128, harmonics=4,
        classifier_lr=3e-4, standardize_step=50)
    for key in ("batch", "candidates", "timesteps", "ddim_steps", "policy_lr", "weight_decay"):
        if asdict(cfg)[key] != control_report["config"][key]:
            raise ValueError(f"Control mismatch: {key}")
    if control_report["seeds"]["policy"] != 17:
        raise ValueError("Control policy seed mismatch")
    data = CubeDistribution(cfg, metadata["reference_mass"], metadata["width"],
        continuous=True, reference_sharpness=metadata["reference_sharpness"])
    initial = Denoiser(cfg); initial.load_state_dict(saved["model"]); initial.eval()
    output.mkdir(parents=True, exist_ok=False)
    report = {"state": "building_classifier_panels", "source": str(source.resolve()),
        "control": str(control.resolve()), "config": asdict(cfg),
        "seeds": {"train": 730017, "validation": 740017, "test": 750017,
            "classifier": 17, "policy": 17, "endpoint": 690017, "monitor": 700017},
        "scope": "Toy Fourier MLP, not production H4. Truth versus actual frozen generator; oracle uses configured mixture, not actual generator density.",
        "primary": "Independent paired frozen-classifier reward gain after 300 DGPO steps with velocity MSE coefficient 1; oracle reward and full mode TV secondary",
        "decision": "Positive learned-gain CI means absorption. No positive gain with a still-improving validation BCE is inconclusive, not reward-interface failure.",
        "classifier_fit": "5000 steps, balanced BCE, best validation checkpoint; test never selects model",
        "panels": {"train": cfg.train_events, "validation": cfg.validation_events, "test": cfg.test_events}}
    import wandb
    wb = wandb.init(project="dgpo-toy", mode=wandb_mode, dir=str(output.resolve()),
        name="Does learned reward transfer? | Fourier cube | V MSE 1 | oracle control",
        group="Conditional cube transport", config=report,
        tags=["frozen-classifier", "low-coverage", "matched-policy"])
    report["wandb"] = {"id": wb.id, "mode": wandb_mode, "directory": wb.dir}
    atomic_json(output / "report.json", report)
    try:
        with (output / "progress.jsonl").open("w") as log:
            def emit(row):
                log.write(json.dumps(row, allow_nan=False) + "\n"); log.flush()
                print(json.dumps(row, allow_nan=False), flush=True)
                wb.log(flatten_metrics(row, row["phase"] + "/"))
            panels = {name: make_panel(initial, data, cfg, n, report["seeds"][name])
                for name, n in report["panels"].items()}
            atomic_checkpoint(output / "panels.pt", panels)
            report["state"] = "fitting_classifier"; atomic_json(output / "report.json", report)
            critic, fit = fit_classifier(cfg, data, panels["train"], panels["validation"],
                17, True, emit, model_class=CubeFourierClassifier)
            report["fit"] = fit
            report["classifier_test"] = classifier_metrics(critic, panels["test"], data)
            report["fit_budget_exhausted_at_best"] = fit["selected_step"] == cfg.classifier_steps
            atomic_checkpoint(output / "classifier.pt", {"model": critic.state_dict(),
                "config": asdict(cfg), "fit": fit, "source": report["source"]})
            base = evaluate(initial, critic, data, cfg, 690017)
            report["initial"] = base[3]
            report["state"] = "training_dgpo"; atomic_json(output / "report.json", report)
            def save(step, model, opt, rng, history):
                report["active_step"] = step
                if step == 1 or step % 50 == 0:
                    stats = evaluate(model, critic, data, cfg, 700017)[3]
                    emit({"phase": "physical", "step": step, **stats})
                    atomic_checkpoint(output / "policy.pt", {"model": model.state_dict(),
                        "optimizer": opt.state_dict(), "rng": rng.get_state(), "history": history,
                        "step": step, "config": asdict(cfg), "velocity_coefficient": 1.,
                        "seed": 17, "monitor_seed": 710017})
                atomic_json(output / "report.json", report)
            model, _ = policy_train("dgpo", initial, critic, data, cfg, 17, 710017,
                emit, save, velocity_coefficient=1.)
            final = evaluate(model, critic, data, cfg, 690017)
            oracle_saved = torch.load(control / "velocity_mse_1.pt", map_location="cpu", weights_only=True)
            oracle_model = Denoiser(cfg); oracle_model.load_state_dict(oracle_saved["model"])
            comparator = evaluate(oracle_model.eval(), critic, data, cfg, 690017)
            for name, result in (("learned_arm", final), ("oracle_control", comparator)):
                report[name] = {"endpoint": result[3]}
                for i, metric in enumerate(("learned_reward", "oracle_reward", "preferred_mass")):
                    report[name][metric + "_gain"] = paired_gain(result[i], base[i])
                    report[name][metric + "_gain_by_condition"] = paired_bins(result[i], base[i])
            report["learned_minus_oracle"] = {name: paired_gain(final[i], comparator[i])
                for i, name in enumerate(("learned_reward", "oracle_reward", "preferred_mass"))}
            atomic_checkpoint(output / "evaluation.pt", {"initial": base[:3],
                "learned": final[:3], "oracle_control": comparator[:3]})
            report["source_weights_unchanged"] = all(torch.equal(v, saved["model"][k]) for k, v in initial.state_dict().items())
            report["state"] = "completed"
    except BaseException as exc:
        report.update(state="failed_or_interrupted", error=repr(exc)); raise
    finally:
        atomic_json(output / "report.json", report)
        wb.summary.update(flatten_metrics(report)); wb.finish(exit_code=0 if report["state"] == "completed" else 1)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("artifacts/dgpo_toy/cube_difficulty_sharp4_v1/pretrain"))
    parser.add_argument("--control", type=Path, default=Path("artifacts/dgpo_toy/cube_difficulty_sharp4_v1/policy"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wandb-mode", choices=("offline", "online", "disabled"), default="offline")
    args = parser.parse_args(); torch.set_num_threads(1); run(**vars(args))
