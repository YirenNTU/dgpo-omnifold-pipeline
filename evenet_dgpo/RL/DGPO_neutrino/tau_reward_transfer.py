"""Paired, fixed-head measurements around native conditional-tau DGPO updates.

This observer neither fits a classifier nor constructs an optimizer/reference.
Every measurement regenerates *all* K candidates with the cycle's fixed
validation identities and isolated generation seed. Unweighted Cij is a
guardrail, not a loss or a criterion for selecting the actor checkpoint.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist


RELATIVE_STEPS = (0, 1, 5, 10, 20, 35, 50)
MECHANISM_RELATIVE_STEPS = (0, 1, 5, 10, 20, 35, 50, 100, 150, 200, 300, 400, 500)
ERROR_KEYS = ("error", "diagonal_error", "offdiagonal_error", "nn_error")


def _normalized_weights(weights: np.ndarray, events: int) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (events,) or not np.isfinite(weights).all():
        raise ValueError("Expected one finite base weight per validation event")
    if (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Tau transfer requires nonnegative, positive-total base weights")
    return weights / weights.sum()


def matrix_errors(delta: np.ndarray) -> dict[str, float]:
    """Frobenius total/diagonal/offdiagonal errors and absolute nn error."""
    delta = np.asarray(delta, dtype=np.float64).reshape(3, 3)
    return {
        "error": float(np.linalg.norm(delta)),
        "diagonal_error": float(np.linalg.norm(np.diag(delta))),
        "offdiagonal_error": float(np.linalg.norm(delta[~np.eye(3, dtype=bool)])),
        "nn_error": float(abs(delta[2, 2])),
    }


def reward_statistics(rewards: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    """Base-weighted statistics; each event divides its mass equally among K."""
    rewards = np.asarray(rewards, dtype=np.float64)
    if rewards.ndim != 2 or min(rewards.shape) < 1 or not np.isfinite(rewards).all():
        raise ValueError("Expected finite validation rewards with shape (events, K)")
    w = _normalized_weights(weights, len(rewards))
    mean = float(w @ rewards.mean(axis=1))
    centered = np.exp(rewards - rewards.max(axis=1, keepdims=True))
    centered /= centered.sum(axis=1, keepdims=True)
    # Median over all candidate scores, with event mass distributed equally.
    order = np.argsort(rewards.ravel(), kind="stable")
    candidate_mass = np.repeat(w / rewards.shape[1], rewards.shape[1])[order]
    median_index = min(int(np.searchsorted(np.cumsum(candidate_mass), .5)), len(order) - 1)
    return {
        "mean": mean,
        "best_of_k": float(w @ rewards.max(axis=1)),
        "worst_of_k": float(w @ rewards.min(axis=1)),
        "median": float(rewards.ravel()[order[median_index]]),
        "within_condition_std": float(w @ rewards.std(axis=1)),
        "within_condition_ess_fraction": float(w @ (1 / (rewards.shape[1] * (centered**2).sum(axis=1)))),
        "global_std": float(np.sqrt(w @ ((rewards - mean)**2).mean(axis=1))),
    }


def paired_statistics(before_reward: np.ndarray, after_reward: np.ndarray,
                      before_cij: np.ndarray, after_cij: np.ndarray,
                      truth_cij: np.ndarray, weights: np.ndarray, *,
                      replicates: int = 300, seed: int = 42) -> dict[str, Any]:
    """Resample conditions, using the SAME multiplicities for reward and Cij.

    The truth target is resampled together with both models. K candidates are
    averaged inside each event, never treated as independent observations.
    One replicate is allocated at a time, independent of replicate count.
    """
    p, q = [np.asarray(a, dtype=np.float64) for a in (before_reward, after_reward)]
    b, a, t = [np.asarray(x, dtype=np.float64) for x in (before_cij, after_cij, truth_cij)]
    if p.shape != q.shape or p.ndim != 2 or len(p) < 2:
        raise ValueError("Paired rewards must match with shape (events >= 2, K)")
    if any(x.shape != (len(p), 9) for x in (b, a, t)):
        raise ValueError("Cij contributions must match rewards with shape (events, 9)")
    if not all(np.isfinite(x).all() for x in (p, q, b, a, t)):
        raise ValueError("Nonfinite paired measurements; do not silently drop events")
    if type(replicates) is not int or replicates < 2:
        raise ValueError("At least two event-bootstrap replicates required")
    w = _normalized_weights(weights, len(p))
    gain = (q-p).mean(axis=1)
    before_delta, after_delta = b-t, a-t
    before_errors, after_errors = [matrix_errors(w @ x) for x in (before_delta, after_delta)]
    changes = {key: after_errors[key]-before_errors[key] for key in ERROR_KEYS}
    rng = np.random.default_rng(seed)
    draws = np.empty((replicates, 1+len(ERROR_KEYS)), dtype=np.float64)
    support = np.flatnonzero(w > 0)
    for index in range(replicates):
        # Sampling only the positive-base-weight support avoids zero-total
        # replicates, without turning importance weights into draw probability.
        multiplicity = np.bincount(rng.choice(support, len(support), replace=True), minlength=len(w))
        bw = multiplicity * w
        bw /= bw.sum()
        draws[index, 0] = bw @ gain
        old, new = [matrix_errors(bw @ x) for x in (before_delta, after_delta)]
        draws[index, 1:] = [new[key]-old[key] for key in ERROR_KEYS]
    intervals = np.quantile(draws, [.025, .975], axis=0)
    return {
        "reward": {
            "delta_mean": float(w @ gain),
            "delta_lo95": float(intervals[0, 0]),
            "delta_hi95": float(intervals[1, 0]),
            "fraction_improving": float(w @ (gain > 0)),
            "event_fraction_improving": float((gain > 0).mean()),
        },
        "cij": {
            "truth": (w @ t).reshape(3, 3).tolist(),
            "generated": (w @ a).reshape(3, 3).tolist(),
            "delta": (w @ after_delta).reshape(3, 3).tolist(),
            "change": (w @ (a-b)).reshape(3, 3).tolist(),
            **after_errors,
            "baseline_errors": before_errors,
            "error_changes": {key: {
                "mean": changes[key], "lo95": float(intervals[0, i+1]),
                "hi95": float(intervals[1, i+1]),
            } for i, key in enumerate(ERROR_KEYS)},
        },
    }


def _frozen_state(model: torch.nn.Module) -> tuple:
    """Cheap identity/version guard; no serialization or extra copy of weights."""
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Fixed reward/reference parameters must not require gradients")
    return (id(model), tuple((name, id(value), value.data_ptr(), value._version,
                             tuple(value.shape), value.dtype, value.device)
                            for name, value in list(model.named_parameters())+list(model.named_buffers())))


class TauRewardTransferProbe:
    """Observe a frozen head/reference and optionally a common independent judge."""

    def __init__(self, cycle, cfg, *, source_step: int, reward_round: int, ensemble=None):
        self.cycle, self.cfg = cycle, dict(cfg)
        self.source_step, self.reward_round = int(source_step), int(reward_round)
        self.mechanism_protocol = self.cfg.get("mechanism_protocol", False)
        if type(self.mechanism_protocol) is not bool:
            raise ValueError("mechanism_protocol must be an explicit boolean")
        self.relative_steps = tuple(self.cfg.get("relative_steps", RELATIVE_STEPS))
        if self.mechanism_protocol:
            if (not self.relative_steps or any(type(step) is not int for step in self.relative_steps)
                    or self.relative_steps[-1] not in (50, 200, 500)
                    or self.relative_steps != tuple(step for step in MECHANISM_RELATIVE_STEPS
                                                   if step <= self.relative_steps[-1])):
                raise ValueError("Mechanism trajectory requires the predeclared 50/200/500 endpoint prefix")
        elif self.relative_steps != RELATIVE_STEPS:
            raise ValueError("Tau transfer endpoints are predeclared: 0/1/5/10/20/35/50")
        if cycle.world != 16 and not self.cfg.get("allow_cpu_fixture", False):
            raise ValueError("Production tau transfer requires 16 workers")
        if cycle.world == 16 and len(cycle.validation["source_ids"]) != 119002:
            raise ValueError("Use the full 119002-event filtered validation panel")
        if int(cycle.cfg["validation_candidates"]) != 8:
            raise ValueError("Tau transfer uses all eight candidates per condition")
        audit_steps = self.cfg.get("audit_relative_steps", [])
        if len(set(audit_steps)) != len(audit_steps) or any(step not in self.relative_steps for step in audit_steps):
            raise ValueError("Fresh audit endpoints must be unique declared transfer endpoints")
        self.marginal_steps = tuple(self.cfg.get("marginal_relative_steps", [0, self.relative_steps[-1]]))
        if len(set(self.marginal_steps)) != len(self.marginal_steps) or any(step not in self.relative_steps for step in self.marginal_steps):
            raise ValueError("Marginal endpoints must be unique declared transfer endpoints")
        self.denominator_step = int(cycle.reward.denominator_step)
        self.head_state = _frozen_state(cycle.reward.head)
        self.reference_state = _frozen_state(cycle.reference)
        self.ensemble = None
        self.judge_state = None
        if self.mechanism_protocol:
            if ensemble is not None:
                self.ensemble = ensemble
            elif self.cfg.get("ensemble_directory") is not None:
                from RL.DGPO_neutrino.tau_classifier_ensemble_probe import TauClassifierEnsemble
                self.ensemble = TauClassifierEnsemble.load(self.cfg["ensemble_directory"], cycle.reward, cycle.device)
            if self.ensemble is not None:
                self.judge_state = _frozen_state(self.ensemble.judge_head)
        self.output = Path(self.cfg.get("output_directory", cycle.output / "fixed-reward-transfer"))
        self.output.mkdir(parents=True, exist_ok=True)
        self.started = False
        self.last_relative_step = 0
        self.baseline = None
        self.training_reward_baseline = None
        evaluator = "common_independent_judge" if self.ensemble is not None else "installed_training_head"
        endpoint = self.relative_steps[-1]
        self.results = {
            "complete": False, "source_step": self.source_step,
            "reward_round": self.reward_round, "denominator_step": self.denominator_step,
            "fixed_head_and_reference": True, "relative_steps": list(self.relative_steps),
            "primary_endpoint": (f"common independent judge held-out all-sample mean reward gain at +{endpoint}"
                                 if self.ensemble is not None else f"fixed-head held-out all-sample mean reward gain at +{endpoint}"),
            "scope": "Native actor updates; fixed installed classifier and reference; raw samples. Cij is unweighted by classifier ratios.",
            "uncertainty": "Paired condition bootstrap, fixed models and generation noise; not training-seed uncertainty. Positive base-weight support is resampled uniformly.",
            "physics_convention": "Existing fixed-energy tau reconstruction and parquet analyzing powers; matched validation truth.",
            "selection": f"Predeclared +{endpoint} endpoint; Cij guardrails are not actor-selection criteria or a training objective.",
            "measurements": {},
        }
        if self.mechanism_protocol:
            self.results.update(mechanism_protocol=True, reward_arm=self.cfg.get("reward_arm", "inherited"),
                reward_evaluator=evaluator, comparable_across_arms=self.ensemble is not None,
                ensemble_directory=(str(self.cfg["ensemble_directory"]) if self.cfg.get("ensemble_directory") is not None else None),
                reward_comparison_scope=(
                    "The same frozen independent fifth judge evaluates every arm's identical validation conditions and generation noise. Training-head rewards are separate diagnostics."
                    if self.ensemble is not None else
                    "Only this arm's own frozen training head is available. Its reward scale and gains are incomparable across different training-head arms; no ensemble efficacy conclusion is supported."))
        if self.cfg.get("full_trajectory"):
            self.results.update(full_trajectory=dict(self.cfg["full_trajectory"]),
                comparable_across_arms=True,
                reward_comparison_scope="Both methods inherit the exact same training head from the same checkpoint; reward comparison is valid but not independent. Fresh audits and spin closure assess transfer.",
                native_training_directory=self.cfg.get("native_training_directory"),
                scope="Step1920 matched native versus full-DDIM pathwise reward; inherited velocity-MSE reference and AdamW; fixed classifier; raw samples.")
            if self.cfg['full_trajectory'].get('reward_refit') is not None:
                self.results.update(comparable_across_arms=False,
                    primary_endpoint=f"Fresh independent audit and full Cij closure at +{endpoint}",
                    reward_comparison_scope="Fresh ratio-bound arms have different fitted teachers; compare spin closure and independent audits, not raw training-head reward across arms.",
                    scope="One startup balanced truth/step1920 reward refit and matching reference recenter; then frozen teacher/reference and inherited AdamW; full-DDIM pathwise updates, no exact KL or hard trust region.")
            if self.cfg['full_trajectory'].get('classifier_kl') is not None:
                self.results.update(scope='Frozen truth/step1920 reward; coefficient1 current/step1920 classifier KL through full DDIM; feature-space approximation; velocity penalty disabled.')
            if self.cfg['full_trajectory'].get('hard_trust_region') is not None:
                self.results.update(hard_trust_region=dict(self.cfg['full_trajectory']['hard_trust_region']),
                    scope='Coefficient1 classifier-KL objective with transactional hard acceptance gate on held-out estimated feature-space mean KL to fixed step1920; finite critic bias is not certified.')
        self._assert_frozen()

    def _assert_frozen(self):
        if self.cycle.reward.round_id != self.reward_round or int(self.cycle.reward.denominator_step) != self.denominator_step:
            raise ValueError("Reward round/denominator changed during fixed-head tau transfer")
        if _frozen_state(self.cycle.reward.head) != self.head_state:
            raise ValueError("Classifier weights changed during fixed-head tau transfer")
        if _frozen_state(self.cycle.reference) != self.reference_state:
            raise ValueError("Reference weights changed during fixed-head tau transfer")
        if self.ensemble is not None and _frozen_state(self.ensemble.judge_head) != self.judge_state:
            raise ValueError("Common independent judge weights changed during mechanism trajectory")

    def _require_all(self, condition: bool, message: str):
        value = torch.tensor(int(condition), device=self.cycle.device, dtype=torch.int32)
        if self.cycle.world > 1:
            dist.all_reduce(value, op=dist.ReduceOp.MIN)
        if not bool(value.item()):
            raise ValueError(message)

    def start(self):
        if self.started:
            raise ValueError("Tau transfer baseline has already been measured")
        self._assert_frozen()
        self._measure(0)
        self.started = True

    def before_update(self, batch):
        """Data disjointness is enforced by the launcher/cycle, not changed here."""
        if not self.started:
            raise ValueError("Measure the fixed-head baseline before native updates")
        self._assert_frozen()

    def stop_at_trust_boundary(self, global_step, metrics):
        """Additional cold endpoint at the last ACCEPTED policy, no clock tick."""
        self._assert_frozen()
        relative = int(global_step)-self.source_step
        if relative != self.last_relative_step:
            raise ValueError('Trust stop must retain the last accepted policy clock')
        if relative > 0:
            saved = self.cfg.get('audit_relative_steps', [])
            try:
                self.cfg['audit_relative_steps'] = sorted(set(saved) | {relative})
                self._measure(relative, metrics)
            finally:
                self.cfg['audit_relative_steps'] = saved
        self.results.update(complete=False, stopped_at_trust_boundary=True,
            accepted_updates=relative, terminal_policy_step=global_step,
            conclusion='trust_radius_exhausted_before_predeclared_endpoint',
            endpoint_scope='Additional cold audit at the restored incumbent; predeclared +50/+200/+500 endpoint not reached')
        if self.cycle.rank == 0:
            target = self.output/'report.json'
            pending = target.with_suffix('.json.tmp')
            pending.write_text(json.dumps(self.results, indent=2, allow_nan=False)+'\n')
            pending.replace(target)
            self.cycle.emit({'tau/classifier_trust/stopped':1, 'tau/classifier_trust/accepted_updates':relative}, global_step)
        if self.cycle.world > 1:
            dist.barrier()

    def after_update(self, global_step: int, metrics: dict, reward_round: int):
        if not self.started:
            raise ValueError("Tau transfer has not started")
        self._require_all(reward_round == self.reward_round,
                          "Reward round changed inside fixed-head tau transfer")
        self._require_all(float(metrics.get("train/optimizer_step_ran", 0)) >= .5,
                          "Skipped optimizer update cannot count as a native applied step")
        relative = int(global_step)-self.source_step
        if relative != self.last_relative_step+1 or relative > self.relative_steps[-1]:
            raise ValueError(f"Tau transfer requires consecutive applied updates through +{self.relative_steps[-1]}")
        self.last_relative_step = relative
        if relative not in self.relative_steps:
            return
        if self.cfg.get("gradient_trace_enabled", False):
            error = float(metrics.get("gradient_transfer/reconstruction_actual/relative_error", float("inf")))
            self._require_all(np.isfinite(error) and error <= float(self.cfg.get("reconstruction_tolerance", 1e-4)),
                              "Native gradient reconstruction failed at a declared tau-transfer endpoint")
        self._assert_frozen()
        self._measure(relative, metrics)

    def _generate(self, folder):
        from RL.DGPO_neutrino.checkpoint_transfer import isolated_evaluation
        # The inner cycle generator restores a single top-level training flag;
        # this outer guard restores mixed frozen/eval submodule modes as well.
        with isolated_evaluation(self.cycle.actor, int(self.cycle.cfg["validation_seed"])+self.cycle.rank):
            return self.cycle.generate(self.cycle.validation, folder, 8,
                                       self.cycle.cfg["validation_seed"], True)

    def _score_common_judge(self, generated, candidate_folder, folder, relative):
        """Rescore saved K8 candidates with the frozen fifth head on every rank."""
        if self.ensemble is None:
            return generated, {}
        # The cycle scorer mutates only rewards. Other arrays and the saved
        # candidate files are shared, preserving the exact same candidates.
        judged = dict(generated)
        if "rewards" in generated:
            judged["rewards"] = generated["rewards"].copy()
        with self.ensemble.temporary_reward(self.cycle.reward, "judge"):
            metrics = self.cycle._score_transfer_head(judged, candidate_folder,
                folder / "independent-judge-scores", self.source_step+relative, "common_judge")
        self._assert_frozen()
        return judged, metrics or {}

    def _measure(self, relative: int, training_metrics: dict | None = None):
        from RL.DGPO_neutrino.conditional_tau_cycle import cij_terms
        panel = self.cycle.validation
        folder = self.output / f"step-{relative:02d}"
        g = self._generate(folder / "candidates")
        replay_error = None
        if relative == 0:
            replay = self._generate(folder / "no-update-replay")
            error = (max(float(np.max(np.abs(replay[key]-g[key]))) for key in ("rewards", "deltas"))
                     if self.cycle.rank == 0 else 0.)
            max_error = torch.tensor(error, device=self.cycle.device, dtype=torch.float64)
            if self.cycle.world > 1:
                dist.all_reduce(max_error, op=dist.ReduceOp.MAX)
            replay_error = float(max_error.item())
            self._require_all(np.isfinite(replay_error) and replay_error <= 1e-6,
                              "No-update fixed-noise tau generation replay failed")
            del replay
        audit = (self.cycle.transfer_audit(g, folder / "fresh_audit", self.source_step+relative)
                 if relative in self.cfg.get("audit_relative_steps", []) else {})
        judged, judge_metrics = self._score_common_judge(g, folder / "candidates", folder, relative)
        if self.cycle.rank == 0:
            folder.mkdir(parents=True, exist_ok=True)
            truth = cij_terms(panel, np.arange(len(panel["source_ids"])), panel["truth_deltas"])
            rewards, generated, weights = judged["rewards"], g["cij"], panel["event_weight"]
            if (rewards.shape != (len(panel["source_ids"]), 8)
                    or g["deltas"].shape != (len(rewards), 8, 2, 2)
                    or not np.isfinite(g["deltas"]).all()):
                raise ValueError("Expected finite complete K8 tau candidate measurements")
            if relative == 0:
                self.baseline = (rewards.copy(), generated.copy())
                self.training_reward_baseline = g["rewards"].copy()
            result = paired_statistics(self.baseline[0], rewards, self.baseline[1], generated,
                truth, weights, replicates=int(self.cfg.get("bootstrap_replicates", 300)),
                seed=int(self.cfg.get("bootstrap_seed", 42)))
            result["reward"].update(reward_statistics(rewards, weights))
            if self.mechanism_protocol:
                result["reward_evaluator"] = self.results["reward_evaluator"]
                if self.ensemble is not None:
                    result["common_judge"] = judge_metrics
                    result["training_reward"] = paired_statistics(self.training_reward_baseline, g["rewards"],
                        self.baseline[1], generated, truth, weights,
                        replicates=int(self.cfg.get("bootstrap_replicates", 300)),
                        seed=int(self.cfg.get("bootstrap_seed", 42)))["reward"]
                    result["training_reward"].update(reward_statistics(g["rewards"], weights))
            result.update(relative_step=relative, policy_step=self.source_step+relative,
                          events=len(rewards), candidates=rewards.shape[1])
            if audit:
                result["fresh_audit"] = audit
            if replay_error is not None:
                result["no_update_replay_max_abs_error"] = replay_error
            gradient = {key.removeprefix("gradient_transfer/").replace("h4", "reward"): float(value)
                        for key, value in (training_metrics or {}).items()
                        if key.startswith("gradient_transfer/")}
            if gradient:
                result["gradient"] = {key: value if np.isfinite(value) else None for key, value in gradient.items()}
            extra_arrays = ({"training_reward": g["rewards"], "judge_reward": rewards}
                            if self.ensemble is not None else {})
            np.savez_compressed(folder / "measurements.npz", source_ids=panel["source_ids"],
                reward=rewards, deltas=g["deltas"], generated_cij=generated,
                truth_cij=truth, weight=weights, **extra_arrays)
            self.results["measurements"][str(relative)] = result
            payload = {"tau/transfer/relative_step": relative,
                       "tau/transfer/policy_step": self.source_step+relative,
                       "tau/transfer/reward_round": self.reward_round,
                       "tau/transfer/events": len(rewards), "tau/transfer/candidates": rewards.shape[1]}
            payload.update({f"tau/transfer/reward/{key}": value for key, value in result["reward"].items()})
            if self.ensemble is not None:
                payload.update({f"tau/transfer/training_reward/{key}": value for key, value in result["training_reward"].items()})
                payload["tau/transfer/common_judge/enabled"] = 1
            payload.update({f"tau/transfer/fresh_audit/{key}": value for key, value in audit.items()})
            payload.update({f"tau/transfer/gradient/{key}": value for key, value in gradient.items()})
            if replay_error is not None:
                payload["tau/transfer/no_update_replay_max_abs_error"] = replay_error
            if relative in self.marginal_steps:
                from RL.DGPO_neutrino.diagnostics.tau_marginals import monitor
                marginal_metrics, plots = monitor(panel, g["deltas"], self.output / "step-00" / "measurements.npz", folder)
                result["marginals"] = dict(metrics=marginal_metrics,
                                           plots={name: str(path) for name, path in plots.items()})
                payload.update({key.replace("tau/marginal/", "tau/transfer/marginal/"): value
                                for key, value in marginal_metrics.items()})
                if plots:
                    import wandb
                    payload.update({f"tau/transfer/marginal/plots/{name}": wandb.Image(str(path))
                                    for name, path in plots.items()})
            for key in ERROR_KEYS:
                payload[f"tau/transfer/cij/{key}"] = result["cij"][key]
                payload.update({f"tau/transfer/cij/change/{key}/{name}": value
                                for name, value in result["cij"]["error_changes"][key].items()})
            for name in ("truth", "generated", "delta", "change"):
                for i, left in enumerate("krn"):
                    for j, right in enumerate("krn"):
                        payload[f"tau/transfer/cij/{name}/{left}{right}"] = result["cij"][name][i][j]
            if relative == self.relative_steps[-1]:
                r = result["reward"]
                self.results.update(complete=True, conclusion=(
                    "supports_local_transfer" if r["delta_lo95"] > 0 else
                    "supports_local_failure" if r["delta_hi95"] < 0 else "unresolved_at_this_precision"))
            target = self.output / "report.json"
            temporary = target.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(self.results, indent=2, allow_nan=False)+"\n")
            temporary.replace(target)
            self.cycle.emit(payload, self.source_step+relative)
        if self.cycle.world > 1:
            dist.barrier()
