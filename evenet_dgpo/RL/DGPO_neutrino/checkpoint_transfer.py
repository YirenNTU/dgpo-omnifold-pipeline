"""Opt-in paired measurements around the *native* DGPO optimizer updates.

No optimizer, gradient, reward or reference is constructed here. The trainer
supplies its existing raw policy sampler/scorer. All K candidates are scored.
"""
from __future__ import annotations

from contextlib import contextmanager
from hashlib import blake2b
import json
import logging
from pathlib import Path
import random
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor
import torch.distributed as dist

from RL.DGPO_neutrino.local_rng import seeded_torch_rng, model_cuda_devices
from RL.DGPO_neutrino.rewards import get_event_valid_mask

_log = logging.getLogger(__name__)


@contextmanager
def isolated_evaluation(model: torch.nn.Module, seed: int):
    """Restore RNG and every submodule mode, including mixed frozen/eval modes."""
    modes = [(module, module.training) for module in model.modules()]
    py_state, np_state = random.getstate(), np.random.get_state()
    try:
        with seeded_torch_rng(seed, model_cuda_devices(model)):
            random.seed(seed)
            np.random.seed(seed % (2**32))
            model.eval()
            with torch.no_grad():
                yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        for module, training in modes:
            module.training = training


def select_rows(batch: dict[str, Any], rows: Any) -> dict[str, Any]:
    size = len(batch["x"])
    return {
        key: value[rows].detach().cpu().clone()
        if isinstance(value, Tensor) and value.ndim and len(value) == size
        else value.detach().cpu().clone() if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def event_fingerprints(batch: dict[str, Any]) -> list[str]:
    # Native data preprocessing does not retain source IDs. Hash the original
    # visible conditions (not candidates, class labels or learned features).
    keys = ("x", "x_mask", "conditions", "conditions_mask")
    arrays = [batch[key].detach().cpu().contiguous().numpy() for key in keys]
    return [blake2b(b"".join(a[i].tobytes() for a in arrays), digest_size=16).hexdigest()
            for i in range(len(batch["x"]))]


def paired_statistics(before: np.ndarray, after: np.ndarray, *, replicates: int,
                      seed: int) -> dict[str, float]:
    """Input is (event, candidate); resample events, never K pseudo-replicates."""
    before, after = np.asarray(before, dtype=np.float64), np.asarray(after, dtype=np.float64)
    if before.shape != after.shape or before.ndim != 2 or len(before) < 2:
        raise ValueError("paired scores must have matching (events>=2, K) shapes")
    if not (np.isfinite(before).all() and np.isfinite(after).all()):
        raise ValueError("Non-finite paired rewards; do not drop events silently")
    delta = (after - before).mean(axis=1)
    rng = np.random.default_rng(seed)
    # Bounded allocation even with larger panels.
    draws = np.empty(replicates)
    for start in range(0, replicates, 32):
        count = min(32, replicates - start)
        draws[start:start + count] = delta[rng.integers(len(delta), size=(count, len(delta)))].mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {
        "events": float(len(delta)), "candidates_per_event": float(before.shape[1]),
        "all_sample_mean": float(after.mean()), "baseline_all_sample_mean": float(before.mean()),
        "delta_mean": float(delta.mean()), "delta_lo95": float(lo), "delta_hi95": float(hi),
        "delta_se": float(delta.std(ddof=1) / np.sqrt(len(delta))),
        "delta_median": float(np.median(delta)), "fraction_improving": float((delta > 0).mean()),
        "candidate_reward_std": float(after.std()),
        "within_event_reward_std_mean": float(after.std(axis=1).mean()),
    }


def gather_objects(value: Any, world_size: int) -> list[Any]:
    if world_size == 1:
        return [value]
    values: list[Any] = [None] * world_size
    dist.all_gather_object(values, value)
    return values


class CheckpointTransferProbe:
    def __init__(self, cfg: Any, *, model: torch.nn.Module, score: Callable,
                 validation_shard: Any, rank: int, world_size: int,
                 source_step: int, reward_round: int, optimizer: Any,
                 checkpoint: dict[str, Any], log: Callable):
        self.cfg, self.model, self.score, self.log = cfg, model, score, log
        self.rank, self.world_size, self.source_step = rank, world_size, source_step
        self.reward_round = reward_round
        self.steps = tuple(int(x) for x in cfg["relative_steps"])
        if self.steps not in ((0, 1, 5, 20), (0, 1, 5, 20, 35, 50)):
            raise ValueError("Use predeclared 20-update or 50-update measurement endpoints")
        if source_step != int(cfg["source_step"]) or reward_round != int(cfg["source_reward_round"]):
            raise ValueError("Restored source step/reward round does not match pinned source")
        if world_size != int(cfg["world_size"]):
            raise ValueError("Probe worker count differs from pinned protocol")
        saved_opt = checkpoint["dgpo_optimizer_state_dict"]
        saved_groups = saved_opt["optimizer"]["param_groups"]
        live_groups = optimizer.param_groups
        if len(saved_groups) != len(live_groups):
            raise ValueError("Optimizer group count changed at restore")
        for old, new in zip(saved_groups, live_groups, strict=True):
            for key in ("lr", "weight_decay", "betas", "eps", "group_name"):
                if old.get(key) != new.get(key):
                    raise ValueError(f"Native resume changed optimizer {key}: {old.get(key)} != {new.get(key)}")
        if optimizer.scheduler.last_epoch != saved_opt["scheduler"]["last_epoch"]:
            raise ValueError("Native scheduler counter was reset")
        self.output = Path(str(cfg["output_directory"]))
        self.output.mkdir(parents=True, exist_ok=True)
        self.baselines: dict[str, np.ndarray] = {}
        self.panels: dict[str, list[dict[str, Any]]] = {}
        self.ids: dict[str, list[str]] = {}
        self.results: dict[str, Any] = {"source_step": source_step, "reward_round": reward_round,
            "primary_endpoint": self.steps[-1], "status": "running", "measurements": {}, "gradient_traces": {}, "panel_layout": {},
            "interpretation": "Fixed-reward transfer, not fresh-classifier closure",
            "uncertainty": "Paired event bootstrap; one optimization trajectory, not training-seed uncertainty",
            "rng_limit": "Original checkpoint lacks saved data-iterator/RNG state; native full-state continuation, not bitwise replay"}
        batches, count = [], 0
        wanted, batch_size = int(cfg["events_per_rank"]), int(cfg["evaluation_batch_size"])
        paired_directory = cfg.get("paired_panel_directory")
        if validation_shard is None and not paired_directory:
            raise ValueError("Independent validation shard required")
        with isolated_evaluation(model, int(cfg["noise_seed"]) + rank):
            if paired_directory:
                saved = torch.load(Path(paired_directory) / f"heldout_panel_rank{rank:02d}.pt",
                                   map_location="cpu", weights_only=False)
                iterator = saved["batches"]
            else:
                iterator = validation_shard.iter_torch_batches(batch_size=batch_size, prefetch_batches=1)
            for batch in iterator:
                valid = get_event_valid_mask(batch, len(batch["x"]), torch.device("cpu"), torch.float32) > 0
                rows = valid.nonzero(as_tuple=True)[0][:wanted - count]
                if len(rows):
                    batches.append(select_rows(batch, rows))
                    count += len(rows)
                if count == wanted:
                    break
        counts = gather_objects(count, world_size)
        if any(n != wanted for n in counts):
            raise ValueError(f"Insufficient validation events per rank: {counts}")
        self._install_panel("heldout", batches)
        self.heldout_ids = set(x for ids in gather_objects(self.ids["heldout"], world_size) for x in ids)
        if len(self.heldout_ids) != wanted * world_size:
            raise ValueError("Duplicate visible-condition identities in held-out panel")
        self.measure("heldout", relative_step=0)
        self.update_cache = None
        if cfg.get("conditioning_ablation", False):
            from .conditioning_probe import PairedUpdateCache
            self.update_cache = PairedUpdateCache(cfg, rank, source_step)

    def training_iterator(self, shard, loader_cfg):
        if self.update_cache is not None:
            return self.update_cache.iterator(shard, loader_cfg)
        return iter(shard.iter_torch_batches(**loader_cfg))

    def _install_panel(self, name: str, batches: list[dict[str, Any]]) -> None:
        self.panels[name] = batches
        self.ids[name] = [key for batch in batches for key in event_fingerprints(batch)]
        self.results["panel_layout"][name] = {
            "rank0_batch_sizes": [len(batch["x"]) for batch in batches],
            "world_size": self.world_size,
        }
        torch.save({"batches": batches, "event_ids": self.ids[name]},
                   self.output / f"{name}_panel_rank{self.rank:02d}.pt")

    def before_update(self, batch: dict[str, Any]) -> None:
        overlaps = len(self.heldout_ids.intersection(event_fingerprints(batch)))
        if any(gather_objects(overlaps, self.world_size)):
            raise ValueError("Held-out event identity overlaps policy update data")
        if "update_batch" not in self.panels:
            valid = get_event_valid_mask(batch, len(batch["x"]), torch.device("cpu"), torch.float32) > 0
            filtered = select_rows(batch, valid)
            size = int(self.cfg["evaluation_batch_size"])
            batches = [select_rows(filtered, slice(i, i + size)) for i in range(0, len(filtered["x"]), size)]
            counts = gather_objects(len(filtered["x"]), self.world_size)
            if any(n < 1 for n in counts):
                raise ValueError("First update batch has an empty valid shard")
            self._install_panel("update_batch", batches)
            self.measure("update_batch", relative_step=0)
        if self.update_cache is not None:
            self.update_cache.before_update(batch)

    def _score_panel(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        rewards, samples = [], []
        for index, batch in enumerate(self.panels[name]):
            if self.rank == 0 and (index % 4 == 0 or index + 1 == len(self.panels[name])):
                _log.info("[DGPO/transfer] %s: evaluation batch %s/%s, %s events/rank",
                          name, index + 1, len(self.panels[name]), len(batch["x"]))
            seed = int(self.cfg["noise_seed"]) + 100000 * (name == "update_batch") + self.rank * 1000 + index
            with isolated_evaluation(self.model, seed):
                candidates, reward = self.score(batch)
                if not torch.isfinite(candidates).all() or not torch.isfinite(reward).all():
                    raise ValueError("Non-finite diagnostic samples or reward")
                rewards.append(reward.detach().double().cpu().numpy().T.copy())
                samples.append(candidates.detach().cpu().numpy().swapaxes(0, 1).copy())
        return np.concatenate(rewards), np.concatenate(samples)

    def measure(self, name: str, *, relative_step: int) -> None:
        _log.info("[DGPO/transfer] +%s: scoring %s with fixed identities/noise", relative_step, name)
        rewards, samples = self._score_panel(name)
        if relative_step == 0:
            replay_r, replay_x = self._score_panel(name)
            replay_error = max(float(np.max(np.abs(replay_r - rewards))), float(np.max(np.abs(replay_x - samples))))
            replay_errors = gather_objects(replay_error, self.world_size)
            if max(replay_errors) > 1e-6:
                raise ValueError(f"No-update paired replay failed: max errors={replay_errors}")
            self.baselines[name] = rewards.copy()
            if self.cfg.get("paired_panel_directory"):
                target = Path(self.cfg["paired_panel_directory"]) / f"{name}_step00_rank{self.rank:02d}.npz"
                with np.load(target) as control:
                    if not np.array_equal(control["event_ids"], np.asarray(self.ids[name])):
                        raise ValueError("Conditioning arms have different initial event identities")
                    initial_error = max(float(np.max(np.abs(control["rewards"] - rewards))),
                                        float(np.max(np.abs(control["candidates"] - samples))))
                initial_errors = gather_objects(initial_error, self.world_size)
                if max(initial_errors) > 1e-6:
                    raise ValueError(f"Conditioning arms changed the initial sampling function: {initial_errors}")
        np.savez_compressed(self.output / f"{name}_step{relative_step:02d}_rank{self.rank:02d}.npz",
                            event_ids=np.asarray(self.ids[name]), rewards=rewards, candidates=samples)
        collected = gather_objects((self.baselines[name], rewards), self.world_size)
        if self.rank == 0:
            stats = paired_statistics(np.concatenate([x[0] for x in collected]),
                np.concatenate([x[1] for x in collected]),
                replicates=int(self.cfg["bootstrap_replicates"]), seed=int(self.cfg["bootstrap_seed"]))
            if relative_step == 0:
                stats["no_update_replay_max_abs_error"] = max(replay_errors)
                if self.cfg.get("paired_panel_directory"):
                    stats["matched_control_initial_max_abs_error"] = max(initial_errors)
            key = f"{name}/+{relative_step}"
            self.results["measurements"][key] = stats
            if name == "heldout" and relative_step == self.steps[-1]:
                self.results["conclusion"] = ("supports_local_transfer" if stats["delta_lo95"] > 0 else
                    "supports_local_failure" if stats["delta_hi95"] < 0 else "unresolved_at_this_precision")
            self._write_report()
            self.log({"checkpoint_transfer/relative_step": relative_step,
                      **{f"checkpoint_transfer/{name}/{k}": v for k, v in stats.items()}},
                     self.source_step + relative_step)
            _log.info("[DGPO/transfer] %s: delta=%g CI95=[%g,%g]", key,
                      stats["delta_mean"], stats["delta_lo95"], stats["delta_hi95"])

    def after_update(self, global_step: int, metrics: dict[str, Any], reward_round: int) -> None:
        if reward_round != self.reward_round:
            raise ValueError("Reward/reference round changed inside fixed-reward diagnostic")
        if float(metrics.get("train/optimizer_step_ran", 0)) < .5:
            raise ValueError("Skipped native update: cannot label this an applied-step probe")
        relative = global_step - self.source_step
        if relative not in self.steps:
            return
        trace = {key: float(value) for key, value in metrics.items() if key.startswith("gradient_transfer/")}
        error = trace.get("gradient_transfer/reconstruction_actual/relative_error", float("inf"))
        errors = gather_objects(error, self.world_size)
        if not all(np.isfinite(e) and e <= float(self.cfg["reconstruction_tolerance"]) for e in errors):
            raise ValueError(f"Gradient reconstruction failed: {errors}")
        if self.rank == 0:
            self.results["gradient_traces"][str(relative)] = {
                key: value if np.isfinite(value) else None for key, value in trace.items()}
        for name in self.panels:
            self.measure(name, relative_step=relative)
        if self.rank == 0 and relative == self.steps[-1]:
            self.results["status"] = "complete"
            self._write_report()

    def _write_report(self) -> None:
        target = self.output / "report.json"
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.results, indent=2, allow_nan=False) + "\n")
        tmp.replace(target)
