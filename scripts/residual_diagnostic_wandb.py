"""Rank-zero-only logging for the standalone residual diagnostic.

No model watching or tensor uploads. Each fit owns a custom x-axis; W&B's
transport step always advances normally. No existing training run is resumed.
"""
from __future__ import annotations

import json
import logging
import math
from numbers import Real
from pathlib import Path
import uuid


def numeric_leaves(data, prefix=""):
    out = {}
    for key, value in data.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(numeric_leaves(value, name))
        elif isinstance(value, Real) and math.isfinite(float(value)):
            out[name] = float(value)
    return out


def validate_wandb(settings):
    cfg = dict(settings or {})
    cfg.setdefault("enabled", False)
    cfg.setdefault("log_every_n_steps", 10)
    if type(cfg["enabled"]) is not bool:
        raise ValueError("wandb.enabled must be boolean")
    if type(cfg["log_every_n_steps"]) is not int or cfg["log_every_n_steps"] < 1:
        raise ValueError("wandb.log_every_n_steps must be positive")
    if cfg["enabled"]:
        for key in ("entity", "project", "name"):
            if not isinstance(cfg.get(key), str) or not cfg[key].strip():
                raise ValueError(f"wandb.{key} is required")
    forbidden = {"id", "run_id", "resume", "api_key", "mode"} & set(cfg)
    if forbidden:
        raise ValueError(f"Diagnostic W&B uses a fresh online run and existing credentials; remove {sorted(forbidden)}")
    return cfg


class DiagnosticWandb:
    def __init__(self, cfg, *, rank=0, device=None, sdk=None):
        self.cfg = cfg
        self.settings = validate_wandb(cfg.get("wandb"))
        self.rank, self.device, self.sdk = rank, device, sdk
        self.root = Path(cfg["output_dir"])
        self.run = None
        self.failed = False
        self.axes = {}
        self.event_index = 0

    def _status(self, **fields):
        # Local recovery pointer; never stores credentials or data tensors.
        with (self.root / "wandb_status.json").open("w") as f:
            json.dump({"run_id": getattr(self.run, "id", None),
                       "url": getattr(self.run, "url", None), **fields}, f, indent=2)

    def start(self):
        error = None
        if self.rank == 0 and self.settings["enabled"]:
            try:
                if self.sdk is None:
                    import wandb
                    self.sdk = wandb
                wb = self.settings
                run_id = uuid.uuid4().hex[:12]
                metadata = {key: self.cfg[key] for key in (
                    "policy_checkpoint", "policy_sha256", "expected_policy_step", "training_seeds",
                    "generation_seed", "pool_events", "workers", "diagnostic_iterations",
                    "control_steps", "control_evaluate_every", "cap_quantile",
                    "crossfit_repeat_arms", "crossfit_fold_arms", "run_controls", "physics_bins",
                ) if key in self.cfg}
                adaptive = self.cfg.get("runtime", {}).get("dgpo", {}).get("adaptive_omnifold", {})
                metadata["classifier_recalibration"] = adaptive.get("recalibration", {})
                metadata["outer_split_seed"] = adaptive.get("single_pool_split_seed")
                self.run = self.sdk.init(entity=wb["entity"], project=wb["project"],
                    name=f"{wb['name']}-{run_id[:6]}", id=run_id, resume="never", mode="online",
                    job_type="residual-weight-diagnostic", group="residual-weight-diagnostic",
                    tags=["diagnostic", "10pct", "fixed-policy", "no-policy-updates"],
                    dir=str(self.root), save_code=False,
                    config=metadata)
                if self.run is not None and self.run.id != run_id:
                    # Some SDK versions reuse an already active process run.
                    # Never log to or finish that unrelated run.
                    self.run = None
                    raise RuntimeError("W&B reused another run instead of creating the diagnostic")
                if self.run is None or getattr(self.run, "disabled", False) or getattr(self.run, "offline", False):
                    raise RuntimeError("Expected an online W&B diagnostic run")
                self.run.summary.update({"diagnostic/status": "running", "diagnostic/policy_updates": 0,
                    "diagnostic/reward_installs": 0,
                    "diagnostic/metric_semantics": "Classifier diagnostics on a fixed policy, not DGPO improvement."})
                self._status(status="running")
                print(f"[residual-diagnostic] W&B: {self.run.url}", flush=True)
            except Exception as exc:
                error = exc
                logging.exception("W&B initialization failed; stop before costly diagnostic work")
        # Every rank receives initialization failure before entering fitting
        # collectives, avoiding a rank-zero-only exception/NCCL hang.
        import torch
        import torch.distributed as dist
        failed = bool(error)
        if dist.is_available() and dist.is_initialized():
            flag = torch.tensor([int(failed)], device=self.device, dtype=torch.int32)
            dist.broadcast(flag, src=0)
            failed = bool(flag.item())
        if failed:
            if self.run is not None:
                try:
                    self.run.finish(exit_code=1)
                except Exception:
                    pass
            raise RuntimeError("W&B diagnostic initialization failed; check rank-zero logs and credentials") from error

    def _guard(self, function):
        if self.run is None or self.failed:
            return
        try:
            function()
        except Exception:
            # Do not let a logging-only exception strand other ranks in NCCL.
            self.failed = True
            logging.exception("W&B logging failed; continuing diagnostic with local files")
            try:
                self._status(status="logging_failed_local_results_preserved")
            except Exception:
                logging.exception("Could not write local W&B status")

    def emit(self, prefix, step, values, *, axis="fit_step"):
        def write():
            coordinate = f"{prefix}/{axis}"
            previous = self.axes.get(coordinate)
            if previous is not None and step < previous:
                raise ValueError(f"Diagnostic axis went backwards: {coordinate}")
            if previous is None:
                self.run.define_metric(coordinate, hidden=True)
                self.run.define_metric(f"{prefix}/*", step_metric=coordinate, step_sync=False)
            self.axes[coordinate] = step
            payload = {f"{prefix}/{k}": v for k, v in numeric_leaves(values).items()}
            payload[coordinate] = int(step)
            self.event_index += 1
            payload["diagnostic/event_index"] = self.event_index
            # Never pass a local classifier step as wandb.log(step=...).
            self.run.log(payload)
        self._guard(write)

    @staticmethod
    def _protocol_prefix(base, protocol):
        return base if not protocol else f"{base}/{protocol}"

    def production(self, seed, row, *, protocol=None):
        fold = int(row.get("fold", 0))
        if fold < 1:
            return  # Ensemble/iteration decisions have their own stream.
        repeat = int(row.get("repeat", 1))
        values = {k: row[k] for k in ("training_loss", "gradient_norm", "gradient_clipped") if k in row}
        if row.get("validation_evaluated"):
            values.update({k: row[k] for k in ("validation_loss", "validation_auc", "validation_balanced_accuracy") if k in row})
        prefix = self._protocol_prefix("production", protocol)
        member = (
            f"f{fold}"
            if protocol is None and repeat == 1
            else f"r{repeat}/f{fold}"
        )
        self.emit(
            f"{prefix}/s{seed}/i{int(row['iteration'])}/{member}",
            int(row["step"]),
            values,
        )

    def restored(self, seed, iteration, fold, report, *, repeat=1, protocol=None):
        # Restored-best metrics never overwrite the last live-fit point.
        prefix = self._protocol_prefix("restored", protocol)
        member = (
            f"f{fold}"
            if protocol is None and repeat == 1
            else f"r{repeat}/f{fold}"
        )
        self.emit(
            f"{prefix}/s{seed}/{member}",
            iteration,
            report,
            axis="iteration",
        )

    def iteration(self, seed, iteration, report, *, protocol=None):
        prefix = self._protocol_prefix("stack", protocol)
        self.emit(f"{prefix}/s{seed}", iteration, report, axis="iteration")

    def control_progress(
        self, seed, iteration, fold, arm, row, *, repeat=1, protocol=None
    ):
        step = int(row["step"])
        if step % self.settings["log_every_n_steps"] == 0 or step == self.cfg["control_steps"]:
            prefix = self._protocol_prefix("control", protocol)
            member = (
                f"f{fold}"
                if protocol is None and repeat == 1
                else f"r{repeat}/f{fold}"
            )
            self.emit(
                f"{prefix}/s{seed}/i{iteration}/{member}/{arm}",
                step,
                {
                    k: row[k]
                    for k in ("training_loss", "gradient_norm", "gradient_clipped")
                    if k in row
                },
            )

    def control_evaluation(
        self, seed, iteration, fold, arm, row, *, repeat=1, protocol=None
    ):
        # Do not re-emit the training progress carried in a scoring record.
        prefix = self._protocol_prefix("control", protocol)
        member = (
            f"f{fold}"
            if protocol is None and repeat == 1
            else f"r{repeat}/f{fold}"
        )
        self.emit(
            f"{prefix}/s{seed}/i{iteration}/{member}/{arm}",
            int(row["step"]),
            {
                k: row[k]
                for k in ("own_objective", "original_weights_same_samples")
            },
        )

    def finish(self, exit_code):
        if self.run is None:
            return
        try:
            status = "failed" if exit_code else (
                "completed" if (self.root / "report.json").is_file() else "partial_results_uploaded")
            self.run.summary.update({"diagnostic/status": status,
                                     "diagnostic/logging_failed": self.failed})
            # Small report files only. No pools, model states, per-event logits,
            # config secrets or raw datasets are uploaded as artifacts.
            for name in ("summary.json", "report.json"):
                path = self.root / name
                if path.is_file():
                    self.run.save(str(path), base_path=str(self.root), policy="now")
            self.run.finish(exit_code=exit_code)
            self._status(status=status, logging_failed=self.failed)
        except Exception:
            logging.exception("W&B final sync failed; local diagnostic results remain available")
            try:
                self._status(status="final_sync_failed_local_results_preserved")
            except Exception:
                logging.exception("Could not write local W&B final status")
