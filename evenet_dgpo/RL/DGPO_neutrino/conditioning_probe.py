"""Opt-in late-checkpoint migration and paired inputs for conditioning ablation."""
from __future__ import annotations

import copy
from pathlib import Path
import random
import numpy as np
import torch

PREFIX = "TruthGeneration.visible_conditioning.reward_probe."


def extend_optimizer_for_probe(model, optimizer, checkpoint, *, branch_name="reward_probe"):
    """Insert only new probe IDs; preserve all existing moments/groups/clocks.

    This is not a generic architecture migration. It accepts exactly one added
    child in the existing visible_conditioning group; every shared tensor must
    already equal its serialized raw weight. No silent weights-only fallback.
    """
    if branch_name not in ("reward_probe", "token_readout"):
        raise ValueError("Unsupported conditioning branch migration")
    prefix = "TruthGeneration.visible_conditioning." + branch_name + "."
    source = {k.removeprefix("model."): v for k, v in checkpoint["state_dict"].items()}
    target = model.state_dict()
    if any(k.startswith(prefix) for k in source):
        raise ValueError("conditioning screen must start before the new probe is trained")
    added = [k for k in target if k not in source]
    if not added or any(not k.startswith(prefix) for k in added):
        raise ValueError("Only the conditioning probe may be added to the saved policy")
    for name, tensor in target.items():
        if name.startswith(prefix):
            continue
        if name not in source or not torch.equal(tensor.detach().cpu(), source[name].detach().cpu()):
            raise ValueError(f"Shared raw checkpoint tensor was not restored exactly: {name}")
    branch = getattr(model.TruthGeneration.visible_conditioning, branch_name)
    if torch.count_nonzero(branch.output.weight) or torch.count_nonzero(branch.output.bias):
        raise ValueError("New probe must start at exactly zero output")
    saved = checkpoint["dgpo_optimizer_state_dict"]
    old_groups = saved["optimizer"]["param_groups"]
    if [g.get("group_name") for g in old_groups] != [g.get("group_name") for g in optimizer.param_groups]:
        raise ValueError("Conditioning probe may not add/reorder optimizer groups")
    names = {id(p): n for n, p in model.named_parameters()}
    old_ids = [i for g in old_groups for i in g["params"]]
    if len(set(old_ids)) != len(old_ids):
        raise ValueError("Duplicated source optimizer parameter IDs")
    next_id = max(old_ids, default=-1) + 1
    groups, new_count = [], 0
    for old, live in zip(old_groups, optimizer.param_groups, strict=True):
        ids, index = [], 0
        for p in live["params"]:
            name = names[id(p)]
            if name.startswith(prefix):
                if old["group_name"] != "visible_conditioning":
                    raise ValueError("Probe parameters must inherit the visible_conditioning LR/schedule")
                ids.append(next_id)
                next_id += 1
                new_count += 1
            else:
                if index >= len(old["params"]):
                    raise ValueError("Unexpected extra shared optimizer parameter")
                pid = old["params"][index]
                moment = saved["optimizer"]["state"].get(pid, {})
                for key in ("exp_avg", "exp_avg_sq"):
                    if key in moment and moment[key].shape != p.shape:
                        raise ValueError(f"Saved optimizer moment shape mismatch: {name}/{key}")
                ids.append(pid)
                index += 1
        if index != len(old["params"]):
            raise ValueError("A shared optimizer parameter was removed")
        groups.append({**old, "params": ids})
    if new_count != len(list(branch.parameters())):
        raise ValueError("Not every probe parameter belongs to the optimizer")
    # State tensors and reference payloads are reused unmodified; only ID lists grow.
    migrated = {**saved, "optimizer": {**saved["optimizer"], "param_groups": groups}}
    return {**checkpoint, "dgpo_optimizer_state_dict": migrated}, new_count


class PairedUpdateCache:
    """A writes real native update batches; B/C read those exact batches.

    Each update resets sampling RNG independently of evaluation or model
    construction. The original checkpoint did not save its iterator/RNG, so
    this pairs the NEW experiment, not the old interrupted trajectory.
    """
    def __init__(self, cfg, rank, source_step):
        self.directory = Path(cfg["update_cache_directory"])
        self.mode = cfg["update_cache_mode"]
        self.steps = int(cfg["relative_steps"][-1])
        self.seed = int(cfg["update_seed"])
        self.rank, self.source_step, self.count = rank, source_step, 0
        if self.mode not in ("write", "read"):
            raise ValueError("update cache mode must be write/read")
        self.directory.mkdir(parents=True, exist_ok=True)

    def path(self, index):
        return self.directory / f"update{index:03d}_rank{self.rank:02d}.pt"

    def iterator(self, shard, loader_cfg):
        if self.mode == "write":
            return iter(shard.iter_torch_batches(**loader_cfg))
        def saved():
            for i in range(1, self.steps + 1):
                yield torch.load(self.path(i), weights_only=False, map_location="cpu")
        if self.count:
            raise ValueError("Replay iterator exhausted before the declared endpoint")
        return iter(saved())

    def before_update(self, batch):
        self.count += 1
        if self.count > self.steps:
            raise ValueError("Conditioning probe exceeded its update budget")
        if self.mode == "write":
            cpu = {k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v)
                   for k, v in batch.items()}
            path = self.path(self.count)
            if path.exists():
                raise ValueError("Use a new experiment directory; update cache already exists")
            torch.save(cpu, path)
        seed = self.seed + 100003 * self.count + self.rank
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
