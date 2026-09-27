#!/usr/bin/env python3
"""Matched raw-policy DDIM coverage: legacy20 versus stable-v 20/100/200."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from diagnose_h4_ratio_tail import unpack
from diagnose_h4_spike_coverage import angles, audit_ddim_steps, coverage, load_raw_state
from diagnose_h4_topology_resolution import quantiles, reconstruct, w1

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evenet_dgpo"))
SCHEMA = "h4-ddim-coverage-v1"
ARMS = {"legacy20": ("legacy", 20), "stable20": ("stable_v", 20),
        "stable100": ("stable_v", 100), "stable200": ("stable_v", 200)}
PAIRS = (("legacy20", "stable20"), ("stable20", "stable100"),
         ("stable100", "stable200"), ("stable20", "stable200"))
RUN_NAME = "Does DDIM limit coverage? | raw step-1110 | stable inversion and 20/100/200 steps"
SOURCE_LABELS = {"arm": "step1110", "weights": "raw_state_dict_only"}
CDF_GRID = np.r_[0., np.geomspace(1e-8, np.pi, 257)]
OWNED_FILES = ("COMPLETE", "manifest.json", "runtime.yaml", "panel.pt", "report.json",
               "cdf.png", "initial_noise.pt", *(f"{arm}.pt" for arm in ARMS))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prepare_output(output, source):
    """Reruns replace only named diagnostic outputs, never source/user files."""
    output, source = Path(output).resolve(), Path(source).resolve()
    if (output in (Path(output.anchor), Path.home(), ROOT) or output == source
            or output in source.parents or source in output.parents):
        raise ValueError("Output must be separate from the source and not a workspace/home root")
    if any((output / name).is_dir() for name in OWNED_FILES):
        raise ValueError("A diagnostic output filename is occupied by a directory")
    output.mkdir(parents=True, exist_ok=True)
    replaced = []
    for name in OWNED_FILES:
        path = output / name
        if path.is_file() or path.is_symlink():
            path.unlink()
            replaced.append(name)
    return output, replaced


def load_source(directory, workers, batch_size):
    import yaml
    directory = Path(directory).resolve()
    if not (directory / "COMPLETE").is_file():
        raise ValueError("Source coverage replay is incomplete")
    meta = json.loads((directory / "manifest.json").read_text())
    # h4cov1 predates the arm/weights labels. It supplies a panel and runtime,
    # not candidates: every arm is regenerated from a strictly verified raw
    # step-1110 state in worker(). Missing old labels are not contrary evidence,
    # and must not be retroactively filled in as historical verification.
    expected = dict(K=128, ddim_steps=20, policy_updates=0, classifier_fits=0,
                    workers=workers, batch_size=batch_size)
    expected.update({key: value for key, value in SOURCE_LABELS.items() if key in meta})
    mismatches = [f"{key}: saved={meta[key]!r}, expected={value!r}" if key in meta
                  else f"{key}: missing, expected={value!r}"
                  for key, value in expected.items() if key not in meta or meta[key] != value]
    mismatches.extend(f"{key}: missing required source metadata"
                      for key in ("events", "seed", "checkpoint") if meta.get(key) is None)
    if mismatches:
        raise ValueError("Source coverage metadata mismatch:\n  " + "\n  ".join(mismatches))
    missing_labels = [key for key in SOURCE_LABELS if key not in meta]
    if missing_labels:
        print("[source compatibility] Legacy manifest has no " + ", ".join(missing_labels)
              + "; retaining unknown historical labels. Every new arm loads raw state_dict "
                "and verifies checkpoint global_step=1110 before sampling.", flush=True)
    panel = torch.load(directory / "panel.pt", map_location="cpu", weights_only=True)
    n = len(panel["truth"])
    if n != meta["events"] or n < workers or n < 2 or panel["truth"].shape != (n, 4):
        raise ValueError("Invalid source panel size/target shape")
    for key in ("condition", "test_rows", "pool_rows"):
        if len(panel[key]) != n:
            raise ValueError(f"Unaligned panel: {key}")
    for key in ("test_rows", "pool_rows"):
        if panel[key].ndim != 1 or len(panel[key].unique()) != n:
            raise ValueError(f"Duplicate or malformed panel identities: {key}")
    if not torch.isfinite(panel["condition"]).all() or not torch.isfinite(panel["truth"]).all():
        raise ValueError("Nonfinite source panel")
    # Check packing and truth geometry before consuming GPU time.
    angles(unpack(dict(test_condition=panel["condition"], packing_spec=panel["packing_spec"])), panel["truth"])
    raw = yaml.safe_load((directory / "runtime.yaml").read_text())
    checkpoint = raw["options"]["Training"]["model_checkpoint_load_path"]
    runtime_steps = audit_ddim_steps(raw)
    if checkpoint != meta["checkpoint"] or runtime_steps != 20:
        raise ValueError("Source runtime disagrees with checkpoint/sampler provenance: "
                         f"runtime checkpoint={checkpoint!r}, manifest checkpoint={meta['checkpoint']!r}; "
                         f"runtime DDIM steps={runtime_steps}, expected=20")
    raw.setdefault("compat", {}).update(backend="dgpo-evenet", repo_root=str(ROOT))
    return meta, panel, raw


def make_noise_bank(n, k, batch_size, device, seed):
    """One independent stream; preserve old sequential-chain/batch draw shapes."""
    generator = torch.Generator(device=device).manual_seed(seed)
    bank = torch.empty(n, k, 2, 2, dtype=torch.float32)
    for start in range(0, n, batch_size):
        size = min(batch_size, n - start)
        bank[start:start + size] = torch.stack([
            torch.randn((size, 2, 2), device=device, dtype=torch.float32, generator=generator)
            for _ in range(k)]).permute(1, 0, 2, 3).cpu()
    return bank


def make_replay_sampler(noise, mode, progress=None):
    from evenet.utilities.diffusion_sampler import DDIMSampler

    class ReplayDDIMSampler(DDIMSampler):
        def __init__(self):
            super().__init__(noise.device, x0_mode=mode)
            self.draws_used = 0

        def prior_sde(self, dimensions):
            if self.draws_used >= len(noise) or tuple(dimensions) != tuple(noise.shape[1:]):
                raise ValueError("Replay noise exhausted or chain/batch shape changed")
            if progress is not None and self.draws_used % 16 == 0:
                progress(self.draws_used)
            x = noise[self.draws_used].clone()
            self.draws_used += 1
            return x

    return ReplayDDIMSampler()


def merge_parts(parts, events, k, value_key):
    """Scatter strided rank shards back into the original event order."""
    result = torch.empty(events, k, 2, 2) if value_key == "noise" else torch.empty(events, k, 4)
    seen = []
    for part in parts:
        ids = part["positions"]
        if ids.ndim != 1 or ids.dtype != torch.int64 or ((ids < 0) | (ids >= events)).any():
            raise ValueError("Invalid rank positions")
        values = part[value_key]
        if values.shape != result[ids].shape or not torch.isfinite(values).all():
            raise ValueError("Invalid rank values")
        result[ids] = values
        seen.extend(ids.tolist())
    if sorted(seen) != list(range(events)):
        raise ValueError("Missing/duplicate panel positions")
    return result


def distribution_metrics(truth, generated, names, include_cdf=False):
    t, g = np.asarray(truth), np.asarray(generated)
    result = {}
    for i, name in enumerate(names):
        a, b = t[:, i], g[:, :, i].reshape(-1)
        entry = dict(truth_quantiles=quantiles(a), generated_quantiles=quantiles(b),
                     w1_radians=w1(a, b, np.ones(len(b))))
        if include_cdf:
            entry["cdf"] = dict(threshold_radians=CDF_GRID.tolist(),
                truth=(np.searchsorted(np.sort(a), CDF_GRID, side="left") / len(a)).tolist(),
                generated=(np.searchsorted(np.sort(b), CDF_GRID, side="left") / len(b)).tolist())
        result[name] = entry
    return result


def analyze_arm(panel, generated):
    if generated.ndim != 3 or generated.shape[0] != len(panel["truth"]) or generated.shape[2] != 4:
        raise ValueError("Invalid candidate shape")
    fields = unpack(dict(test_condition=panel["condition"], packing_spec=panel["packing_spec"]))
    truth_angles = angles(fields, panel["truth"])
    candidate_angles, invalid = [], {}
    for k in range(generated.shape[1]):
        r = reconstruct(fields, generated[:, k])
        candidate_angles.append(np.stack([r["acoplanarity"], r["acollinearity"]], -1))
        for name, count in r["invalid"].items():
            invalid[name] = invalid.get(name, 0) + count
    candidate_angles = np.stack(candidate_angles, 1)
    # Do not cut, project or quietly drop finite out-of-range target coordinates.
    # Report them as a guardrail alongside spherical-direction diagnostics.
    t = np.c_[truth_angles, truth_angles.max(-1)]
    g = np.concatenate([candidate_angles, candidate_angles.max(-1, keepdims=True)], -1)
    result = dict(coverage=coverage(truth_angles, candidate_angles), invalid_direction_inputs=invalid,
        topology=distribution_metrics(t, g, ("acoplanarity", "acollinearity", "joint_radius"), True),
        target_marginals=distribution_metrics(panel["truth"].double().numpy(), generated.double().numpy(),
            ("tau_a_delta_theta", "tau_a_delta_phi", "tau_b_delta_theta", "tau_b_delta_phi")))
    return result, truth_angles, candidate_angles


def paired_comparison(truth, before, after, bootstrap=1000, seed=20260920):
    t, a, b = map(np.asarray, (truth, before, after))
    if a.shape != b.shape or a.ndim != 3 or a.shape[0] != len(t) or a.shape[2] != 2 or t.shape != (len(t), 2):
        raise ValueError("Unaligned paired angles")
    if len(t) < 2 or bootstrap < 20 or not all(np.isfinite(v).all() for v in (t, a, b)):
        raise ValueError("Invalid paired inputs/bootstrap budget")
    indices = np.random.default_rng(seed).integers(len(t), size=(bootstrap, len(t)))
    rows = []
    for threshold in (1e-6, 1e-5, 1e-4):
        for region, axes in (("acoplanarity", [0]), ("acollinearity", [1]), ("joint", [0, 1])):
            target = (t[:, axes] < threshold).all(-1).astype(float)
            aa = (a[:, :, axes] < threshold).all(-1).mean(1)
            bb = (b[:, :, axes] < threshold).all(-1).mean(1)
            delta = bb - aa
            gap_before, gap_after = abs(aa.mean() - target.mean()), abs(bb.mean() - target.mean())
            boot_delta = delta[indices].mean(1)
            boot_truth = target[indices].mean(1)
            boot_gap_change = abs(bb[indices].mean(1) - boot_truth) - abs(aa[indices].mean(1) - boot_truth)
            rows.append(dict(region=region, threshold_radians=threshold,
                after_minus_before_draw_fraction=float(delta.mean()),
                paired_event_se=float(delta.std(ddof=1) / np.sqrt(len(t))),
                delta_ci95=np.quantile(boot_delta, [.025, .975]).tolist(),
                absolute_truth_gap_change=float(gap_after - gap_before),
                absolute_truth_gap_change_ci95=np.quantile(boot_gap_change, [.025, .975]).tolist(),
                fraction_of_baseline_gap_closed=float((gap_before - gap_after) / gap_before) if gap_before else None,
                before_hit_events=int((aa > 0).sum()), after_hit_events=int((bb > 0).sum()),
                events_with_nonzero_paired_difference=int((delta != 0).sum())))
    return rows


def update_comparisons(report, truth_angles, saved_angles, cfg):
    for before, after in PAIRS:
        if before in saved_angles and after in saved_angles:
            report["comparisons"][f"{after}_minus_{before}"] = paired_comparison(
                truth_angles, saved_angles[before], saved_angles[after], cfg["bootstrap"], cfg["bootstrap_seed"])


def log_arm(run, arm, result, report):
    if run is None:
        return
    metrics = {f"arms/{arm}/sampling_seconds_slowest_rank": result["sampling_seconds_slowest_rank"],
               f"arms/{arm}/nfe_per_draw": result["steps"]}
    for row in result["coverage"]["regions"]:
        prefix = f"arms/{arm}/{row['region']}/{row['threshold_radians']:g}"
        metrics.update({f"{prefix}/{key}": value for key, value in row.items()
                        if isinstance(value, (int, float)) and key != "threshold_radians"})
    for group in ("topology", "target_marginals"):
        for name, row in result[group].items():
            metrics[f"arms/{arm}/{group}/{name}/w1_radians"] = row["w1_radians"]
    for key, value in result["invalid_direction_inputs"].items():
        metrics[f"arms/{arm}/invalid/{key}"] = value
    for comparison, rows in report["comparisons"].items():
        for row in rows:
            prefix = f"paired/{comparison}/{row['region']}/{row['threshold_radians']:g}"
            for key in ("after_minus_before_draw_fraction", "paired_event_se", "absolute_truth_gap_change"):
                metrics[f"{prefix}/{key}"] = row[key]
            metrics[f"{prefix}/delta_ci95_lo"], metrics[f"{prefix}/delta_ci95_hi"] = row["delta_ci95"]
    run.log(metrics)
    run.summary.update(metrics)


def plot_cdfs(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for axis, name in zip(axes, ("acoplanarity", "acollinearity", "joint_radius")):
        for index, (arm, result) in enumerate(report["arms"].items()):
            curve = result["topology"][name]["cdf"]
            if index == 0:
                axis.plot(curve["threshold_radians"], curve["truth"], "k--", label="truth")
            axis.plot(curve["threshold_radians"], curve["generated"], label=arm)
        axis.set_xscale("symlog", linthresh=1e-8)
        axis.set(xlabel="Threshold (rad)", ylabel="Fraction below threshold", title=name, ylim=(0, 1))
        axis.axvline(1e-4, color="gray", linewidth=.8)
        axis.legend(fontsize=8)
    fig.savefig(Path(output) / "cdf.png", dpi=150)
    plt.close(fig)


def worker(cfg):
    import ray.train
    import ray.train.torch
    from evenet.control.global_config import global_config
    from RL.DGPO_neutrino.model_utils import build_evenet_on_device, load_normalization_dict
    from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import EventPackingSpec, unpack_event_inputs
    from RL.DGPO_neutrino.sampling import generate_neutrino_candidates

    rank = ray.train.get_context().get_world_rank()
    world = ray.train.get_context().get_world_size()
    if world != cfg["workers"]:
        raise ValueError("Worker count changed")
    device = ray.train.torch.get_device()
    output = Path(cfg["output"])
    run, success = None, False
    try:
        if rank == 0 and cfg["wandb"]:
            import wandb
            run = wandb.init(project="nu2flow-RL", entity="ytchou97-university-of-washington",
                id=cfg["run_id"], resume="allow", name=RUN_NAME, group="H4 sampler coverage",
                tags=["DDIM", "raw-only", "step-1110", "paired-noise", "no-policy-update", "no-classifier-fit"], config=cfg)
            run.summary.update(dict(complete=False, phase="load_raw_checkpoint", attempt=cfg["attempt"]))
        global_config.load_yaml(cfg["runtime"])
        torch.set_float32_matmul_precision(str(global_config.dgpo.get("float32_matmul_precision", "medium")))
        source = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=False)
        policy = build_evenet_on_device(global_config, load_normalization_dict(global_config), device).eval()
        load_raw_state(policy, source, "step1110")
        del source
        policy.requires_grad_(False)
        if int(getattr(policy, "invisible_input_dim", 2)) != 2:
            raise ValueError("This experiment requires the two-coordinate neutrino target")
        panel = torch.load(output / "panel.pt", map_location="cpu", weights_only=True)
        positions = torch.arange(rank, len(panel["truth"]), world)
        spec = EventPackingSpec.from_dict(panel["packing_spec"])
        noise = make_noise_bank(len(positions), cfg["K"], cfg["batch_size"], device, cfg["seed"] + 10000 + rank)
        torch.save(dict(positions=positions, noise=noise), output / f"noise-rank-{rank:03d}.pt")
        report = dict(schema=SCHEMA, config=cfg, complete=False, arms={}, comparisons={},
            checkpoint_global_step=1110, weights="raw_state_dict_only",
            float32_matmul_precision=torch.get_float32_matmul_precision(),
            limitation="Reused exploratory panel. No classifier or policy training, projection, selection or reweighting. "
                       "Bootstrap resamples whole events with all K draws; zero empirical CI is not a support bound. "
                       "Steps are not cost matched. No improvement through 200 steps does not prove model incapacity.")
        saved_angles = {}
        if run:
            run.summary.update({"checkpoint/raw_verified": True, "checkpoint/global_step": 1110,
                                "phase": "sampling", "paired_noise": "explicit_tensor_replay"})
        for arm, (mode, steps) in ARMS.items():
            outputs, elapsed = [], 0.
            for start in range(0, len(positions), cfg["batch_size"]):
                ids = positions[start:start + cfg["batch_size"]]
                batch = unpack_event_inputs(panel["condition"][ids].to(device), spec)
                # Shape/mask only; no truth values enter inference.
                batch["x_invisible"] = torch.zeros(len(ids), 2, 2, device=device)
                batch["x_invisible_mask"] = torch.ones(len(ids), 2, dtype=torch.bool, device=device)

                def progress(done):
                    if rank == 0:
                        print(f"[DDIM {arm}] rank0 batch={start//cfg['batch_size']+1} "
                              f"events_done={start}/{len(positions)} chains_done={done}/{cfg['K']} steps={steps}", flush=True)
                        if run:
                            run.log({"progress/arm": arm, "progress/rank0_events_done": start,
                                     "progress/rank0_batch_chains_done": done, "progress/ddim_steps": steps,
                                     "progress/attempt": cfg["attempt"]})

                sampler = make_replay_sampler(noise[start:start + len(ids)].permute(1, 0, 2, 3).to(device), mode, progress)
                torch.cuda.synchronize(device)
                began = time.perf_counter()
                generated = generate_neutrino_candidates(policy, batch, sampler, K=cfg["K"],
                    num_ddim_steps=steps, device=device, parallel_chains=1)
                torch.cuda.synchronize(device)
                elapsed += time.perf_counter() - began
                if sampler.draws_used != cfg["K"] or generated.shape != (cfg["K"], len(ids), 2, 2):
                    raise ValueError("Candidate/noise replay contract changed")
                if not torch.isfinite(generated).all():
                    raise ValueError(f"Nonfinite sampler output in {arm}; not dropping events")
                outputs.append(generated.permute(1, 0, 2, 3).reshape(len(ids), cfg["K"], 4).cpu())
            torch.save(dict(positions=positions, generated=torch.cat(outputs), sampling_seconds=elapsed),
                       output / f"{arm}-rank-{rank:03d}.pt")
            torch.distributed.barrier()
            if rank == 0:
                if not saved_angles:
                    parts = [torch.load(output / f"noise-rank-{i:03d}.pt", weights_only=True) for i in range(world)]
                    torch.save(merge_parts(parts, cfg["events"], cfg["K"], "noise"), output / "initial_noise.pt")
                parts = [torch.load(output / f"{arm}-rank-{i:03d}.pt", weights_only=True) for i in range(world)]
                generated = merge_parts(parts, cfg["events"], cfg["K"], "generated")
                result, truth_angles, candidate_angles = analyze_arm(panel, generated)
                result.update(x0_mode=mode, steps=steps, sampling_seconds_slowest_rank=max(p["sampling_seconds"] for p in parts),
                              sampling_gpu_seconds_sum=sum(p["sampling_seconds"] for p in parts))
                report["arms"][arm] = result
                saved_angles[arm] = candidate_angles
                update_comparisons(report, truth_angles, saved_angles, cfg)
                torch.save(dict(generated=generated, angles=torch.from_numpy(candidate_angles), x0_mode=mode, steps=steps), output / f"{arm}.pt")
                write_json(output / "report.json", report)
                plot_cdfs(report, output)
                log_arm(run, arm, result, report)
                if run:
                    run.log({"plots/angular_cdf": wandb.Image(str(output / "cdf.png"))})
                primary = next(r for r in result["coverage"]["regions"] if r["region"] == "joint" and r["threshold_radians"] == 1e-4)
                print(f"[DDIM {arm} COMPLETE] joint<1e-4 draw_fraction={primary['generated_draw_fraction']:.8g} "
                      f"truth_fraction={primary['truth_fraction']:.8g} hits={primary['generated_draw_count']}", flush=True)
            # Every rank completes analysis before the next arm; no mixed phases.
            torch.distributed.barrier()
        if rank == 0:
            report["complete"] = True
            write_json(output / "report.json", report)
            if run:
                table = wandb.Table(columns=["arm", "steps", "joint_draw_fraction", "truth_fraction", "draw_hits", "seconds"])
                for arm, result in report["arms"].items():
                    row = next(r for r in result["coverage"]["regions"] if r["region"] == "joint" and r["threshold_radians"] == 1e-4)
                    table.add_data(arm, result["steps"], row["generated_draw_fraction"], row["truth_fraction"],
                                   row["generated_draw_count"], result["sampling_seconds_slowest_rank"])
                run.log({"results/primary_comparison": table})
                artifact = wandb.Artifact(f"{cfg['run_id']}-ddim-coverage", type="diagnostic-report")
                for name in ("report.json", "manifest.json", "cdf.png"):
                    artifact.add_file(str(output / name))
                run.log_artifact(artifact)
                run.summary.update(dict(complete=True, phase="complete"))
            (output / "COMPLETE").write_text(SCHEMA + "\n")
        success = True
    finally:
        if run:
            run.finish(exit_code=0 if success else 1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path,
                        default=Path("/pscratch/sd/y/yiren/Ztautau/h4_spike_coverage_1110"))
    parser.add_argument("--output", type=Path, default=Path("/pscratch/sd/y/yiren/Ztautau/h4_ddim_coverage_1110"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--ray-address", default=os.environ.get("RAY_ADDRESS") or "auto")
    parser.add_argument("--run-id", default="h4ddim1")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args(argv)
    if min(args.workers, args.batch_size) < 1 or args.bootstrap < 20:
        parser.error("Positive sizes and bootstrap>=20 required")
    if int(os.environ.get("SLURM_PROCID", "0")) != 0:
        parser.error("Launch once inside the active allocation, not once per GPU")
    meta, panel, raw = load_source(args.directory, args.workers, args.batch_size)
    checkpoint = Path(meta["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    import yaml
    import ray
    from ray.train import RunConfig, ScalingConfig, FailureConfig
    from ray.train.torch import TorchTrainer
    # Connect before replacing old output. This experiment never starts a
    # misleading one-node fallback when the 16-GPU cluster is unavailable.
    ray.init(address=args.ray_address, runtime_env={"env_vars": {
        "PYTHONPATH": os.pathsep.join([str(ROOT / "evenet_dgpo"), str(ROOT / "scripts"), os.environ.get("PYTHONPATH", "")])}})
    if ray.cluster_resources().get("GPU", 0) < args.workers:
        raise RuntimeError(f"Requires {args.workers} GPUs in the running Ray cluster")
    output, replaced = prepare_output(args.output, args.directory)
    if replaced:
        print("Replaced prior diagnostic outputs only: " + ", ".join(replaced), flush=True)
    runtime = output / "runtime.yaml"
    runtime.write_text(yaml.safe_dump(raw, sort_keys=False))
    torch.save(panel, output / "panel.pt")
    attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    cfg = dict(schema=SCHEMA, source=str(args.directory.resolve()), output=str(output), runtime=str(runtime),
        checkpoint=str(checkpoint), weights="raw_state_dict_only", checkpoint_arm="step1110",
        source_manifest_labels={key: meta.get(key) for key in SOURCE_LABELS},
        source_manifest_missing_labels=[key for key in SOURCE_LABELS if key not in meta],
        events=len(panel["truth"]), K=128, seed=meta["seed"], workers=args.workers, batch_size=args.batch_size,
        arms={arm: dict(x0_mode=mode, steps=steps) for arm, (mode, steps) in ARMS.items()},
        parallel_chains=1, eta=1., time_grid="uniform_t; existing cosine logSNR [-20,20] endpoints",
        initial_noise="explicit tensor replay; seed=source_seed+10000+rank; same batch and chain layout",
        bootstrap=args.bootstrap, bootstrap_seed=20260920, policy_updates=0, classifier_fits=0,
        physics_projection=False, candidate_selection=False, confirmatory=False,
        primary_endpoint="K128 joint draw fraction: acoplanarity AND acollinearity <1e-4 rad",
        primary_contrasts=["stable20_minus_legacy20", "stable200_minus_stable20"],
        source_pool_previously_examined=True, wandb=not args.no_wandb, run_id=args.run_id,
        run_name=RUN_NAME, attempt=attempt)
    write_json(output / "manifest.json", cfg)
    print(f"Raw step1110; {cfg['events']} events; identical K=128 noise; arms={list(ARMS)}. "
          "No policy/classifier training. Total NFE/draw=340 (17x one 20-step replay).", flush=True)
    TorchTrainer(train_loop_per_worker=worker, train_loop_config=cfg,
        scaling_config=ScalingConfig(num_workers=args.workers, use_gpu=True),
        run_config=RunConfig(name=f"ddim-coverage-{attempt}", storage_path=str(output / "ray_results"),
                             failure_config=FailureConfig(max_failures=0))).fit()
    print("FULL REPORT:", output / "report.json", flush=True)


if __name__ == "__main__":
    main()
