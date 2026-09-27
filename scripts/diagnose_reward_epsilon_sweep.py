#!/usr/bin/env python3
"""Replay the v5 calibrated-LOO gradient at several parameter-space radii.

The expensive H4 classifiers are loaded from ``diagnose_reward_interface.py``
artifacts.  No classifier is fitted and no optimizer state is loaded.  Every
signed probe starts from the same c4a91e07 policy and uses common rollout noise.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evenet_dgpo"))
sys.path.insert(0, str(ROOT / "scripts"))

import diagnose_raw_monitor_replay as replay
import diagnose_reward_interface as interface
from ablate_raw_monitor_initialization import clone_state

from RL.DGPO_neutrino.reward_interface import reward_advantage_arms, vector_cosine


REQUIRED_ARTIFACTS = (
    "runtime.yaml",
    "manifest.json",
    "report.json",
    "fixed_k1_pool.pt",
    "fixed_k8_panel.pt",
    "classifier_reward_s20260913_f1.pt",
    "classifier_reward_s20260913_f2.pt",
    "classifier_reward_s20260914_f1.pt",
    "classifier_reward_s20260914_f2.pt",
    "independent_h4_judge.pt",
)


def _load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except RuntimeError as exc:
        if "mmap can only be used" not in str(exc):
            raise
        return torch.load(path, map_location="cpu", weights_only=False)


def _validated_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    cfg = dict(settings)
    required = {
        "source_dir", "output_dir", "expected_source_schema", "expected_policy_step",
        "workers", "cpus_per_worker", "epsilons_rms", "K", "gradient_events",
        "gradient_blocks", "gradient_timesteps", "gradient_seed", "beta",
        "policy_eval_t_min", "policy_eval_t_max", "rollout_seed", "score_batch_size",
        "physics_bins", "response_bins", "response_bootstrap_replicates",
        "response_bootstrap_seed", "wandb",
    }
    missing = sorted(required - set(cfg))
    if missing:
        raise ValueError(f"missing epsilon-sweep settings: {missing}")
    for key in (
        "workers", "cpus_per_worker", "K", "gradient_events", "gradient_blocks",
        "gradient_timesteps", "score_batch_size", "physics_bins", "response_bins",
        "response_bootstrap_replicates",
    ):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["K"] < 2:
        raise ValueError("K must be at least two")
    if cfg["gradient_blocks"] < 2 or cfg["gradient_events"] < cfg["gradient_blocks"]:
        raise ValueError("gradient panel must cover at least two nonempty blocks")
    epsilons = [float(value) for value in cfg["epsilons_rms"]]
    if not epsilons or any(not math.isfinite(value) or value <= 0 for value in epsilons):
        raise ValueError("epsilons_rms must contain finite positive values")
    if epsilons != sorted(set(epsilons)):
        raise ValueError("epsilons_rms must be strictly increasing and unique")
    cfg["epsilons_rms"] = epsilons
    wandb = dict(cfg["wandb"] or {})
    wandb.setdefault("enabled", False)
    if type(wandb["enabled"]) is not bool:
        raise ValueError("wandb.enabled must be boolean")
    if wandb.get("required") and not wandb["enabled"]:
        raise ValueError("wandb.required=true requires wandb.enabled=true")
    cfg["wandb"] = wandb
    return cfg


def prepare(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the immutable v5 artifacts without using a digest contract."""

    cfg = _validated_settings(settings)
    source = Path(cfg["source_dir"]).expanduser().resolve(strict=True)
    paths = {name: (source / name).resolve(strict=True) for name in REQUIRED_ARTIFACTS}
    manifest = json.loads(paths["manifest.json"].read_text())
    report = json.loads(paths["report.json"].read_text())
    if report.get("schema") != cfg["expected_source_schema"]:
        raise ValueError(
            f"source schema mismatch: expected {cfg['expected_source_schema']!r}, "
            f"found {report.get('schema')!r}"
        )
    if int(report.get("policy_global_step", -1)) != int(cfg["expected_policy_step"]):
        raise ValueError("source report policy step does not match the declared anchor")
    if not report.get("fixed_policy") or not report.get("policy_unchanged"):
        raise ValueError("source diagnostic did not certify an unchanged fixed policy")
    members = report.get("classifier_members", [])
    member_order = [
        {"seed": int(row["seed"]), "fold": int(row["fold"])} for row in members
    ]
    if member_order != [
        {"seed": 20260913, "fold": 1}, {"seed": 20260913, "fold": 2},
        {"seed": 20260914, "fold": 1}, {"seed": 20260914, "fold": 2},
    ]:
        raise ValueError(f"unexpected source ensemble: {member_order}")
    temperature = float(report["classifier_calibration"]["primary"]["temperature"])
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("source primary temperature is invalid")
    policy_checkpoint = Path(report["policy_checkpoint"]).resolve(strict=True)
    checkpoint = replay.read_checkpoint(policy_checkpoint)
    if int(checkpoint.get("global_step", -1)) != int(cfg["expected_policy_step"]):
        raise ValueError("live policy checkpoint step changed since v5")
    runtime_path = paths["runtime.yaml"]
    runtime = __import__("yaml").safe_load(runtime_path.read_text())
    training = runtime["options"]["Training"]
    if runtime["dgpo"].get("checkpoint_load_mode") != "weights_only":
        raise ValueError("v5 runtime must load policy weights only")
    if any(training.get("EMA", {}).get(key, False) for key in (
        "replace_model_after_load", "use_for_generation", "use_ema_during_training_eval"
    )):
        raise ValueError("v5 runtime unexpectedly selects EMA policy weights")
    output = Path(cfg["output_dir"]).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output already exists; choose a new directory: {output}")
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("epsilon-sweep output overlaps its source artifact directory")
    cfg.update(
        source_dir=str(source), output_dir=str(output), runtime_path=str(runtime_path),
        policy_checkpoint=str(policy_checkpoint), temperature=temperature,
        member_order=member_order, source_manifest=manifest,
        source_stats={
            str(path): [path.stat().st_size, path.stat().st_mtime_ns]
            for path in paths.values()
        },
    )
    del checkpoint
    return cfg


def verify_sources(cfg: Mapping[str, Any]) -> None:
    for path, expected in cfg["source_stats"].items():
        observed = [Path(path).stat().st_size, Path(path).stat().st_mtime_ns]
        if observed != expected:
            raise RuntimeError(f"source artifact changed during replay: {path}")


def summarize_jsd(physics: Mapping[str, Any]) -> dict[str, float]:
    values = {
        key.removeprefix("val_ztautau/jsd/current/"): float(value)
        for key, value in physics.items()
        if key.startswith("val_ztautau/jsd/current/") and math.isfinite(float(value))
    }
    return {
        "mean": float(np.mean(list(values.values()))) if values else float("nan"),
        "maximum": float(np.max(list(values.values()))) if values else float("nan"),
        "count": float(len(values)),
        **{f"observable/{key}": value for key, value in values.items()},
    }


def decide_epsilon(
    zero: Mapping[str, Any], plus: Mapping[str, Any],
    bootstrap: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """Apply the predeclared local eligibility rule to one +epsilon probe."""

    judge_delta = float(plus["judge_auc_gap"]) - float(zero["judge_auc_gap"])
    response_delta = (
        float(plus["response_mean_abs_bin_offset"])
        - float(zero["response_mean_abs_bin_offset"])
    )
    jsd_delta = float(plus["jsd"]["mean"]) - float(zero["jsd"]["mean"])
    significantly_worse = sorted(
        component for component, row in bootstrap.items()
        if float(row["ci95_low"]) > 0.0
    )
    pass_judge = judge_delta < 0.0
    pass_response = response_delta < 0.0 and not significantly_worse
    pass_jsd = jsd_delta < 0.0
    return {
        "eligible": bool(pass_judge and pass_response and pass_jsd),
        "judge_auc_gap_delta": judge_delta,
        "response_mean_abs_bin_offset_delta": response_delta,
        "physics_mean_jsd_delta": jsd_delta,
        "pass_judge": pass_judge,
        "pass_response": pass_response,
        "pass_physics_jsd": pass_jsd,
        "significantly_worse_response_components": significantly_worse,
    }


def select_epsilon(rows: list[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """Choose the largest radius satisfying every predeclared endpoint."""

    eligible = [row for row in rows if bool(row["decision"]["eligible"])]
    return max(eligible, key=lambda row: float(row["epsilon_rms"]), default=None)


def _worker(cfg: Mapping[str, Any]) -> None:
    import ray.train
    import ray.train.torch
    from evenet.control.global_config import global_config
    from evenet.utilities.diffusion_sampler import DDIMSampler
    from RL.DGPO_neutrino.dgpo_trainer import batch_to_device
    from RL.DGPO_neutrino.diagnostics.ztautau_validation import (
        build_ztautau_validation_metrics,
        collect_ztautau_validation_arrays,
    )
    from RL.DGPO_neutrino.model_utils import (
        load_evenet_model_for_dgpo,
        load_normalization_dict,
    )
    from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import (
        EventPackingSpec,
        EvenetAdapterModelBuilder,
        unpack_event_inputs,
    )
    from RL.DGPO_neutrino.sampling import generate_neutrino_candidates

    context = ray.train.get_context()
    rank, world = context.get_world_rank(), context.get_world_size()
    if world != int(cfg["workers"]):
        raise ValueError(f"expected {cfg['workers']} Ray workers, found {world}")
    device = ray.train.torch.get_device()
    global_config.load_yaml(cfg["runtime_path"])
    torch.set_float32_matmul_precision(
        str(global_config.dgpo.get("float32_matmul_precision", "medium"))
    )
    root = Path(cfg["output_dir"])
    source_root = Path(cfg["source_dir"])
    wandb_run = None
    if rank == 0 and cfg["wandb"].get("enabled"):
        import wandb
        wandb_run = wandb.init(
            entity=cfg["wandb"].get("entity"), project=cfg["wandb"].get("project"),
            name=cfg["wandb"].get("name"), tags=cfg["wandb"].get("tags"),
            job_type="diagnostic", config=interface._jsonable({
                key: value for key, value in cfg.items()
                if key not in {"source_manifest", "source_stats"}
            }),
        )
        wandb_run.summary["phase"] = "load_v5_artifacts"

    panel = _load(source_root / "fixed_k8_panel.pt")
    pool_payload = _load(source_root / "fixed_k1_pool.pt")
    packing_spec = EventPackingSpec.from_dict(pool_payload["packing_spec"])
    if list(panel["member_order"]) != cfg["member_order"]:
        raise ValueError("fixed panel member order does not match report")
    if int(panel["candidates_kb22"].shape[0]) != int(cfg["K"]):
        raise ValueError("fixed panel K does not match epsilon-sweep K")
    total_events = int(panel["packed_event"].shape[0])
    if int(cfg["gradient_events"]) > total_events:
        raise ValueError("gradient_events exceeds the saved v5 panel")
    positions = torch.arange(total_events)[rank::world]
    packed = panel["packed_event"][rank::world]
    truth = panel["truth"][rank::world]
    noise_mask = panel["policy_noise_mask"][rank::world]
    fixed_candidates = panel["candidates_kb22"][:, rank::world].to(device)
    batch = unpack_event_inputs(packed, packing_spec)
    batch["x_invisible"] = truth.reshape(-1, 2, 2)
    batch["x_invisible_mask"] = noise_mask
    batch = batch_to_device(batch, device)

    checkpoint = replay.read_checkpoint(cfg["policy_checkpoint"])
    bundle = load_evenet_model_for_dgpo(
        config=global_config, device=device, checkpoint_path=cfg["policy_checkpoint"]
    )
    policy = bundle.model.eval()
    interface.verify_policy_loaded(policy, checkpoint["state_dict"])
    anchor_model_state = clone_state(policy.state_dict())
    reference_bundle = load_evenet_model_for_dgpo(
        config=global_config, device=device, checkpoint_path=cfg["policy_checkpoint"]
    )
    reference = reference_bundle.model.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    interface.verify_policy_loaded(reference, checkpoint["state_dict"])
    del checkpoint

    member_logits = panel["member_logits_mkb"][:, :, rank::world].mean(0).to(device)
    advantage = reward_advantage_arms(
        member_logits, temperature=float(cfg["temperature"]), raw_tempering=0.75,
    )["calibrated_loo"]
    gradient_report, gradients = interface._gradient_audit(
        cfg={**cfg, "training_seeds": [20260913, 20260914]},
        policy=policy, reference=reference, batch=batch, candidates=fixed_candidates,
        advantage_sets={"primary/calibrated_loo": advantage},
        global_positions=positions, device=device, dtype=next(policy.parameters()).dtype,
    )
    direction = gradients["primary/calibrated_loo"]
    parameters = tuple(p for p in policy.parameters() if p.requires_grad)
    anchor = tuple(p.detach().cpu().clone() for p in parameters)

    recalibration = dict(global_config.dgpo.adaptive_omnifold.recalibration)
    builder_keys = set(inspect.signature(EvenetAdapterModelBuilder).parameters)
    builder = EvenetAdapterModelBuilder(
        config=global_config,
        normalization_dict=load_normalization_dict(global_config),
        checkpoint_path=global_config.reward_config.omnifold.backbone_checkpoint,
        device=device,
        **{key: value for key, value in recalibration.items() if key in builder_keys},
    )
    builder.restore_pretrained_body()
    judge = builder.make_classifier(packing_spec, "epsilon_sweep_judge", reset=True).to(device)
    judge.load_state_dict(_load(source_root / "independent_h4_judge.pt"), strict=True)
    judge.eval()
    for parameter in judge.parameters():
        parameter.requires_grad_(False)
    truth_judge_local = interface._score_local_on_device(
        judge, packed, truth, int(cfg["score_batch_size"])
    ).cpu()
    sampler = DDIMSampler(device=device)

    def restore_anchor() -> None:
        with torch.no_grad():
            for parameter, base in zip(parameters, anchor):
                parameter.copy_(base.to(parameter.device, parameter.dtype))

    def rollout() -> torch.Tensor:
        torch.manual_seed(int(cfg["rollout_seed"]) + rank)
        with torch.no_grad():
            return generate_neutrino_candidates(
                policy, batch, sampler, K=int(cfg["K"]),
                num_ddim_steps=int(global_config.dgpo.num_ddim_steps), device=device,
                parallel_chains=int(global_config.dgpo.get("rollout_parallel_chains", 1)),
            )

    restore_anchor()
    baseline_candidates = rollout()

    def evaluate(label: str, candidates: torch.Tensor, scale: float) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        sample = candidates.permute(1, 0, 2, 3).reshape(len(truth), int(cfg["K"]), 4).cpu()
        gen_logits = interface._score_local_on_device(
            judge, packed, sample, int(cfg["score_batch_size"])
        ).cpu()
        gathered_truth = interface._all_gather_object(truth_judge_local)
        gathered_gen = interface._all_gather_object(gen_logits)
        arrays = collect_ztautau_validation_arrays(
            candidates, baseline_candidates, batch,
            torch.ones(len(truth), device=device, dtype=torch.bool),
        )
        arrays = interface._gather_numpy_dict(arrays)
        if rank != 0:
            return {}, {}
        judge_metrics = interface._classification_metrics(
            torch.cat(gathered_truth), torch.cat(gathered_gen).reshape(-1)
        )
        physics = build_ztautau_validation_metrics(
            arrays, val_k=int(cfg["K"]), tarp_config={"enabled": False},
            metrics_config={"enabled": True, "bins": int(cfg["physics_bins"]), "candidate_index": 0},
            include_images=False,
        )
        response, event_scores = interface.response_matrix_metrics(
            arrays["_tarp_truth"], arrays["_tarp_candidates"], bins=int(cfg["response_bins"])
        )
        return {
            "label": label, "flat_gradient_scale": scale,
            "judge": judge_metrics, "judge_auc_gap": judge_metrics["auc_gap"],
            "physics": physics, "jsd": summarize_jsd(physics), "response": response,
            "response_mean_abs_bin_offset": float(np.mean([
                row["mean_abs_bin_offset"] for row in response.values()
            ])),
        }, event_scores

    zero_result, zero_events = evaluate("zero", baseline_candidates, 0.0)
    rows: list[dict[str, Any]] = []
    for epsilon_index, epsilon in enumerate(cfg["epsilons_rms"]):
        signed: dict[str, Any] = {}
        signed_events: dict[str, dict[str, np.ndarray]] = {}
        for sign, sign_name in ((-1, "minus"), (1, "plus")):
            scale = interface._assign_direction(
                parameters, anchor, direction, sign=sign, epsilon_rms=float(epsilon)
            )
            stepped_gradient_report, stepped_gradients = interface._gradient_audit(
                cfg={**cfg, "training_seeds": [20260913, 20260914]},
                policy=policy, reference=reference, batch=batch, candidates=fixed_candidates,
                advantage_sets={"primary/calibrated_loo": advantage},
                global_positions=positions, device=device,
                dtype=next(policy.parameters()).dtype,
            )
            candidates = rollout()
            result, event_scores = evaluate(sign_name, candidates, scale)
            if rank == 0:
                result["epsilon_rms"] = float(epsilon)
                result["anchor_gradient_cosine"] = vector_cosine(
                    direction, stepped_gradients["primary/calibrated_loo"]
                )
                result["stepped_gradient"] = stepped_gradient_report[
                    "primary/calibrated_loo"
                ]
                signed[sign_name] = result
                signed_events[sign_name] = event_scores
            restore_anchor()
        if rank == 0:
            bootstrap = {
                component: interface._paired_bootstrap_delta(
                    signed_events["plus"][component], zero_events[component],
                    replicates=int(cfg["response_bootstrap_replicates"]),
                    seed=int(cfg["response_bootstrap_seed"]) + 100 * epsilon_index + index,
                )
                for index, component in enumerate(zero_events)
            }
            decision = decide_epsilon(zero_result, signed["plus"], bootstrap)
            row = {
                "epsilon_rms": float(epsilon), "zero": zero_result,
                **signed, "plus_vs_zero_response_bootstrap": bootstrap,
                "decision": decision,
            }
            rows.append(row)
            print(
                f"[epsilon-sweep] eps={epsilon:.1e} eligible={decision['eligible']} "
                f"dAUCgap={decision['judge_auc_gap_delta']:+.6g} "
                f"dResponse={decision['response_mean_abs_bin_offset_delta']:+.6g} "
                f"dJSD={decision['physics_mean_jsd_delta']:+.6g} "
                f"grad_cos={signed['plus']['anchor_gradient_cosine']:.4f}",
                flush=True,
            )
            if wandb_run is not None:
                wandb_run.log({
                    "epsilon/index": epsilon_index,
                    "epsilon/rms": epsilon,
                    "epsilon/eligible": float(decision["eligible"]),
                    "epsilon/plus/judge_auc_gap": signed["plus"]["judge_auc_gap"],
                    "epsilon/plus/response_mean_abs_bin_offset": signed["plus"]["response_mean_abs_bin_offset"],
                    "epsilon/plus/physics_mean_jsd": signed["plus"]["jsd"]["mean"],
                    "epsilon/plus/anchor_gradient_cosine": signed["plus"]["anchor_gradient_cosine"],
                    "epsilon/minus/judge_auc_gap": signed["minus"]["judge_auc_gap"],
                    "epsilon/minus/response_mean_abs_bin_offset": signed["minus"]["response_mean_abs_bin_offset"],
                    "epsilon/minus/physics_mean_jsd": signed["minus"]["jsd"]["mean"],
                }, step=epsilon_index)

    restore_anchor()
    if rank == 0:
        interface.verify_policy_loaded(policy, anchor_model_state)
        eligible = [row for row in rows if row["decision"]["eligible"]]
        # Use the largest radius that passes every predeclared endpoint.  This
        # maximizes measurable policy movement without comparing heterogeneous
        # AUC, response-bin, and JSD units in one arbitrary scalar score.
        selected = select_epsilon(rows)
        report = {
            "schema": "c4a91e07-h4-calibrated-loo-epsilon-sweep-v1",
            "source_schema": cfg["expected_source_schema"],
            "source_dir": cfg["source_dir"],
            "source_wandb_run": cfg["source_manifest"].get("source_wandb_run"),
            "policy_checkpoint": cfg["policy_checkpoint"],
            "policy_global_step": int(cfg["expected_policy_step"]),
            "policy_updates": 0, "classifier_fits": 0, "optimizer_state_resumed": False,
            "reward_arm": "calibrated_loo", "temperature": float(cfg["temperature"]),
            "common_event_panel": True, "common_rollout_noise": True,
            "anchor_gradient": gradient_report["primary/calibrated_loo"],
            "zero": zero_result, "sweep": rows,
            "eligible_epsilons_rms": [row["epsilon_rms"] for row in eligible],
            "selected_epsilon_rms": None if selected is None else selected["epsilon_rms"],
            "decision_scope": (
                "Local paired direction test only. Eligibility requires simultaneous "
                "improvement in independent-judge AUC gap, aggregate response offset, "
                "and mean physics JSD, with no response component significantly worse."
            ),
        }
        verify_sources(cfg)
        replay._exclusive_json(root / "report.json", interface._jsonable(report))
        if wandb_run is not None:
            import wandb
            artifact = wandb.Artifact(
                "c4a91e07-h4-calibrated-loo-epsilon-sweep", type="diagnostic"
            )
            artifact.add_file(str(root / "report.json"))
            wandb_run.log_artifact(artifact)
            wandb_run.summary.update({
                "phase": "complete", "classifier_fits": 0, "policy_updates": 0,
                "eligible_epsilon_count": len(eligible),
                "selected_epsilon_rms": None if selected is None else selected["epsilon_rms"],
            })
            wandb_run.finish()
        print(f"Epsilon-sweep report: {root / 'report.json'}", flush=True)
    ray.train.report({"completed": 1, "classifier_fits": 0, "policy_updates": 0})


def main() -> int:
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text())
    if args.output_dir is not None:
        settings["output_dir"] = str(args.output_dir)
    cfg = prepare(settings)
    print(
        f"Verified v5 artifacts; replaying {len(cfg['epsilons_rms'])} epsilons from "
        f"policy step {cfg['expected_policy_step']} with zero classifier fits.", flush=True,
    )
    if args.check_only:
        return 0
    if cfg["wandb"].get("required"):
        disabled = os.environ.get("WANDB_DISABLED", "").lower() in {
            "1", "true", "yes",
        }
        offline = os.environ.get("WANDB_MODE", "").lower() in {
            "offline", "disabled", "dryrun",
        }
        if disabled or offline:
            raise RuntimeError(
                "this experiment requires live W&B; WANDB_DISABLED/WANDB_MODE "
                "currently disables online logging"
            )
    root = Path(cfg["output_dir"])
    import ray
    from ray.train import FailureConfig, RunConfig, ScalingConfig
    from ray.train.torch import TorchTrainer
    ray_address = os.environ.get("RAY_ADDRESS")
    if not ray_address:
        raise RuntimeError(
            "RAY_ADDRESS is unset; start/source the 16-GPU Ray cluster before "
            "launching the epsilon sweep"
        )
    ray.init(
        address=ray_address,
        runtime_env={"env_vars": {"PYTHONPATH": os.pathsep.join([
            str(ROOT / "evenet_dgpo"), str(ROOT / "scripts"),
            os.environ.get("PYTHONPATH", ""),
        ])}},
    )
    available_gpus = float(ray.cluster_resources().get("GPU", 0) or 0)
    if available_gpus < int(cfg["workers"]):
        raise RuntimeError(
            f"Ray cluster has {available_gpus:g} GPUs; epsilon sweep requires "
            f"{cfg['workers']}"
        )
    root.mkdir(parents=True, exist_ok=False)
    replay._exclusive_json(root / "manifest.json", interface._jsonable(cfg))
    TorchTrainer(
        train_loop_per_worker=_worker, train_loop_config=cfg,
        scaling_config=ScalingConfig(
            num_workers=int(cfg["workers"]), use_gpu=True,
            resources_per_worker={"CPU": int(cfg["cpus_per_worker"]), "GPU": 1},
        ),
        run_config=RunConfig(
            name="c4a91e07-h4-calibrated-loo-epsilon-sweep-v1",
            storage_path=str(root / "ray_results"),
            failure_config=FailureConfig(max_failures=0),
        ),
    ).fit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
