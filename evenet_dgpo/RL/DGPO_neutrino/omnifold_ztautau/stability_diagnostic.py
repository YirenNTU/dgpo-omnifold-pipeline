"""Opt-in classifier diagnostics. Hooks never change activations or gradients.

Probe scores are measurement-only; snapshot files contain private event data.
All ranks must call the public methods in the same order.
"""
from __future__ import annotations

import logging
import math
import random
import statistics
import tempfile
from collections import deque
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

LOG = logging.getLogger(__name__)


def clone_tree(value, *, cpu=False):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone() if cpu else value.detach().clone()
    if isinstance(value, dict):
        return {k: clone_tree(v, cpu=cpu) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(clone_tree(v, cpu=cpu) for v in value)
    return value


def rng_state(device):
    return {"torch": torch.get_rng_state(), "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}


def restore_rng(state, device):
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)


def diagnostic_modules(model):
    """EveNet's shared backbone is intentionally not a registered submodule."""
    modules = dict(model.named_modules())
    backbone = getattr(model, "backbone", None)
    if backbone is not None:
        modules.update({f"backbone.{n}" if n else "backbone": m for n, m in backbone.named_modules()})
    seen, result = set(), {}
    for name, module in modules.items():
        if id(module) not in seen:
            result[name] = module
            seen.add(id(module))
    return result


def diagnostic_buffers(model):
    return {f"{name}.{key}" if name else key: value
            for name, module in diagnostic_modules(model).items()
            for key, value in module.named_buffers(recurse=False)}


class StabilityDiagnostic:
    def __init__(self, model, validation, config, *, rank, world, seed):
        if validation is None:
            raise ValueError("Stability diagnostics require early-stop validation, never the final test pool")
        self.model, self.config = model, config
        self.rank, self.world, self.seed = rank, world, seed
        self.device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        self.probe = []
        for c, z, w in (validation[:3], validation[3:]):
            if len(z) < world:
                raise ValueError("Stability probe requires at least one validation row per rank")
            index = torch.arange(rank, min(len(z), world * config.diagnostic_probe_rows), world, device=z.device)
            self.probe.extend([t[index].to(device=self.device, dtype=dtype) for t in (c, z, w)])
        self.saved = 0
        self.attempted = 0
        self.last_snapshot_step = None
        self.history = deque(maxlen=50)
        self.handles = []
        self.layer_values = {}
        self.metrics = {}
        self.before = None
        self.pre_update = None
        self.representation = None
        if config.representation_diagnostic_enabled:
            from .representation_diagnostic import RepresentationDiagnostic
            self.representation = RepresentationDiagnostic(self, validation)

    def maximum(self, value):
        value = torch.tensor(float(value), dtype=torch.float64, device=self.device)
        if self.world > 1:
            torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.MAX)
        return float(value.item())

    def record(self, key, value):
        self.layer_values.setdefault(key, []).append(value.detach().double())

    def rms(self, name, tensor):
        self.record(name, tensor.detach().double().square().mean().sqrt())

    def watch_gradient(self, name, tensor):
        if tensor.requires_grad:
            self.handles.append(tensor.register_hook(lambda grad: self.rms(name, grad)))

    @torch.no_grad()
    def attention(self, name, module, args, kwargs):
        # Shadow QK calculation only: never request attention weights from the
        # real forward, which would change its SDPA dispatch. Bounded query rows.
        if module.bias_k is not None or module.add_zero_attn:
            return
        q = args[0] if args else kwargs["query"]
        k = args[1] if len(args) > 1 else kwargs["key"]
        if q.ndim != 3:
            return
        if not module.batch_first:
            q, k = q.transpose(0, 1), k.transpose(0, 1)
        q, k = q[:2, :16], k[:2]
        d, h = module.embed_dim, module.num_heads
        wq = module.in_proj_weight[:d] if module.in_proj_weight is not None else module.q_proj_weight
        wk = module.in_proj_weight[d:2*d] if module.in_proj_weight is not None else module.k_proj_weight
        bias = module.in_proj_bias
        q = F.linear(q.float(), wq.float(), None if bias is None else bias[:d].float())
        k = F.linear(k.float(), wk.float(), None if bias is None else bias[d:2*d].float())
        self.rms(f"{name}/projected_q_rms", q)
        self.rms(f"{name}/projected_k_rms", k)
        q = q.reshape(q.shape[0], q.shape[1], h, d // h).transpose(1, 2)
        k = k.reshape(k.shape[0], k.shape[1], h, d // h).transpose(1, 2)
        scores = q @ k.transpose(-1, -2) / math.sqrt(d // h)
        self.record(f"{name}/qk_absmax", scores.abs().amax())
        padding = kwargs.get("key_padding_mask", args[3] if len(args) > 3 else None)
        mask = kwargs.get("attn_mask", args[5] if len(args) > 5 else None)
        if padding is not None:
            padding = padding[:scores.shape[0], None, None, :]
            scores = scores.masked_fill(padding, -torch.inf) if padding.dtype == torch.bool else scores + padding
        if mask is not None:
            if mask.ndim == 2:
                mask = mask[None, None, :scores.shape[-2], :]
            else:
                mask = mask.reshape(-1, h, mask.shape[-2], mask.shape[-1])[:scores.shape[0], :, :scores.shape[-2], :]
            scores = scores.masked_fill(mask, -torch.inf) if mask.dtype == torch.bool else scores + mask
        if kwargs.get("is_causal", False):
            causal = torch.ones(scores.shape[-2:], device=scores.device, dtype=torch.bool).triu(1)
            scores = scores.masked_fill(causal, -torch.inf)
        valid = torch.isfinite(scores).any(dim=-1)
        self.record(f"{name}/all_masked_fraction", (~valid).float().mean())
        if valid.any():
            p = scores[valid].softmax(-1)
            self.record(f"{name}/attention_entropy", -(p * p.clamp_min(1e-30).log()).sum(-1).mean())

    def install_hooks(self):
        modules = diagnostic_modules(self.model)
        for name, module in modules.items():
            scope = name.startswith(("bank.decoder.", "backbone.PET."))
            parent = modules.get(name.rsplit(".", 1)[0], module)
            if not scope or not (name.startswith("bank.decoder.") or any(p.requires_grad for p in parent.parameters())):
                continue
            adapter = module.__class__.__name__ == "Adapter"
            if not (adapter or isinstance(module, (nn.LayerNorm, nn.MultiheadAttention))):
                continue
            def before(mod, args, kwargs, name=name):
                if isinstance(mod, nn.MultiheadAttention):
                    try:
                        self.attention(name, mod, args, kwargs)
                    except Exception:
                        self.record(f"{name}/shadow_diagnostic_error", torch.ones((), device=self.device))
                        LOG.exception("[classifier/stability] QK shadow diagnostic failed for %s", name)
                x = args[0] if args else kwargs.get("query")
                if isinstance(x, torch.Tensor):
                    self.rms(f"{name}/input_rms", x)
                    self.watch_gradient(f"{name}/input_gradient_rms", x)
                    if isinstance(mod, nn.LayerNorm):
                        axes = tuple(range(x.ndim - len(mod.normalized_shape), x.ndim))
                        variance = x.detach().float().var(dim=axes, unbiased=False)
                        self.record(f"{name}/pre_norm_variance_min", variance.amin())
                        self.record(f"{name}/pre_norm_variance_mean", variance.mean())
            def after(mod, args, kwargs, output, name=name, adapter=adapter):
                y = output[0] if isinstance(output, tuple) else output
                if isinstance(y, torch.Tensor):
                    self.rms(f"{name}/output_rms", y)
                    self.watch_gradient(f"{name}/output_gradient_rms", y)
                    if adapter:
                        x = args[0].detach().double()
                        ratio = (y.detach().double() - x).square().mean().sqrt() / x.square().mean().sqrt().clamp_min(1e-30)
                        self.record(f"{name}/residual_input_rms_ratio", ratio)
            self.handles.append(module.register_forward_pre_hook(before, with_kwargs=True))
            self.handles.append(module.register_forward_hook(after, with_kwargs=True))

    def close_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    @torch.no_grad()
    def probe_scores(self):
        state = rng_state(self.device)
        modes = {m: m.training for m in diagnostic_modules(self.model).values()}
        buffers = {name: b.detach().clone() for name, b in diagnostic_buffers(self.model).items()}
        error = None
        outputs = []
        try:
            self.model.eval()
            for c, z, _ in (self.probe[:3], self.probe[3:]):
                outputs.append(self.model(c, z).detach().clone())
        except Exception as exc:
            error = str(exc)
        finally:
            for m, mode in modes.items():
                m.training = mode
            for name, buffer in diagnostic_buffers(self.model).items():
                buffer.copy_(buffers[name])
            restore_rng(state, self.device)
        if self.maximum(error is not None):
            raise RuntimeError(f"Classifier diagnostic probe failed on a rank: {error or 'see other rank logs'}")
        return outputs

    def probe_metrics(self, scores):
        values = []
        for index, logits in enumerate(scores):
            weights = self.probe[index * 3 + 2].reshape(-1).double()
            logits = logits.reshape(-1).double()
            losses = F.softplus(-logits if index == 0 else logits)
            values.extend([(weights * losses).sum(), weights.sum(), logits.sum(), logits.new_tensor(logits.numel())])
        sums = torch.stack(values)
        if self.world > 1:
            torch.distributed.all_reduce(sums)
        return {"bce": float((.5 * (sums[0]/sums[1] + sums[4]/sums[5])).item()),
                "separation": float((sums[2]/sums[3] - sums[6]/sums[7]).item())}

    def begin(self, step, batch, optimizer, microbatch_rows):
        self.step, self.batch, self.optimizer = step, batch, optimizer
        self.microbatch_rows = microbatch_rows
        self.sampled = step == 1 or step % self.config.diagnostic_interval_steps == 0
        if self.representation is not None and step <= 5:
            self.sampled = True
        self.metrics, self.layer_values = {}, {}
        normalizer = getattr(getattr(self.model, 'bank', None), 'topology_standardizer', None)
        if self.sampled and normalizer is not None:
            self.metrics['stability/fourier_output_scale/enabled'] = float(self.config.fourier_output_standardization)
            self.metrics['stability/fourier_output_scale/min'] = float(normalizer.scale.min())
            self.metrics['stability/fourier_output_scale/max'] = float(normalizer.scale.max())
            self.metrics['stability/fourier_output_scale/mean_rms'] = float(normalizer.mean.square().mean().sqrt())
        if self.representation is not None and not self.representation.local_baseline_done:
            self.representation.probe(step - 1)
            self.representation.local_baseline_done = True
        self.pre_update = None
        self.start_rng = rng_state(self.device)
        self.start_modes = {n: m.training for n, m in diagnostic_modules(self.model).items()}
        self.start_buffers = {n: b.detach().clone() for n, b in diagnostic_buffers(self.model).items()}
        self.before = self.probe_scores() if self.sampled else None
        if self.sampled:
            self.install_hooks()
            if self.representation is not None:
                self.representation.install_training_hooks()

    def after_backward(self, loss):
        self.close_hooks()
        grads = [p.grad.detach().double().square().sum() for p in self.model.parameters() if p.grad is not None]
        self.local_norm = float(torch.stack(grads).sum().sqrt().item()) if grads else 0.
        self.local_loss = float(loss.detach().item())
        peak = self.maximum(self.local_norm if math.isfinite(self.local_norm) else math.inf)
        self.metrics["stability/local_gradient_norm_rankmax"] = peak
        self.metrics["stability/local_gradient_norm"] = self.local_norm
        self.metrics["stability/local_training_loss_rankmax"] = self.maximum(self.local_loss if math.isfinite(self.local_loss) else math.inf)
        if self.sampled:
            # Key union tolerates unused layers on a rank; no collective in hooks.
            local = {k: float((torch.stack(v).amin() if k.endswith("_min") else
                              torch.stack(v).amax() if k.endswith("absmax") else torch.stack(v).mean()).item())
                     for k, v in self.layer_values.items()}
            rows = [local]
            if self.world > 1:
                rows = [None] * self.world
                torch.distributed.all_gather_object(rows, local)
            for key in sorted(set().union(*(r.keys() for r in rows))):
                values = [r[key] for r in rows if key in r]
                self.metrics[f"stability/layer/{key}/rankmean"] = sum(values) / len(values)
                self.metrics[f"stability/layer/{key}/rankmax"] = max(values)
                if key.endswith("_min"):
                    self.metrics[f"stability/layer/{key}/rankmin"] = min(values)
        baseline = statistics.median(self.history) if self.history else 0.
        spike = (peak >= self.config.diagnostic_gradient_threshold or
                 (len(self.history) >= 10 and peak > self.config.diagnostic_spike_factor * max(baseline, 1e-12)))
        self.metrics["stability/gradient_spike"] = float(spike)
        if math.isfinite(peak):
            self.history.append(peak)
        if spike:
            self.save("gradient_spike", before_update=True)

    def prepare_update(self):
        armed = bool(self.config.diagnostic_snapshot_dir and self.attempted < self.config.diagnostic_max_snapshots)
        if armed or self.sampled:
            self.pre_update = {
                "parameters": {n: p.detach().clone() for n, p in self.model.named_parameters() if p.requires_grad},
                "optimizer": clone_tree(self.optimizer.state_dict()) if armed else None,
            }

    def after_update(self):
        if self.sampled:
            grouped = {}
            for name, parameter in self.model.named_parameters():
                if name not in self.pre_update["parameters"]:
                    continue
                old_parameter = self.pre_update["parameters"][name].double()
                delta = parameter.detach().double() - old_parameter
                grouped.setdefault(name.rsplit(".", 1)[0], []).append(torch.stack([
                    delta.square().sum(), old_parameter.square().sum(), delta.new_tensor(delta.numel())]))
            for name, values in grouped.items():
                change, scale, count = torch.stack(values).sum(0).tolist()
                self.metrics[f"stability/update/{name}/rms"] = math.sqrt(change / count)
                self.metrics[f"stability/update/{name}/relative_rms"] = math.sqrt(change / max(scale, 1e-30))
            if self.representation is not None:
                for name, p in self.model.named_parameters():
                    if name.startswith('bank.output.') or '.modulation.proj.' in name:
                        self.metrics[f"stability/representation/parameter/{name}/rms"] = float(p.detach().double().square().mean().sqrt())
            after = self.probe_scores()
            old, new = self.probe_metrics(self.before), self.probe_metrics(after)
            for name in old:
                self.metrics[f"stability/probe/{name}_before"] = old[name]
                self.metrics[f"stability/probe/{name}_after"] = new[name]
                self.metrics[f"stability/probe/{name}_delta"] = new[name] - old[name]
            sums = torch.stack([sum((a.double()-b.double()).square().sum() for a, b in zip(after, self.before)),
                                torch.tensor(sum(a.numel() for a in after), device=self.device, dtype=torch.float64)])
            if self.world > 1:
                torch.distributed.all_reduce(sums)
            self.metrics["stability/probe/logit_change_rms"] = float((sums[0]/sums[1]).sqrt().item())
            if not math.isfinite(new["bce"]) or new["bce"] - old["bce"] >= self.config.diagnostic_probe_bce_jump:
                self.save("probe_bce_jump", before_update=False)
        self.metrics["stability/snapshots_saved_local"] = float(self.saved)
        if self.representation is not None and (
            self.step in (10, 50) or self.step % self.config.representation_probe_interval_steps == 0
        ):
            self.representation.probe(self.step)
        self.pre_update = None

    def save(self, reason, *, before_update):
        if (self.attempted >= self.config.diagnostic_max_snapshots or not self.config.diagnostic_snapshot_dir
                or self.last_snapshot_step == self.step):
            return
        if not before_update and self.pre_update is None:
            return
        # Snapshot failures are diagnostic failures, not a reason to alter a fit.
        self.attempted += 1
        self.last_snapshot_step = self.step
        try:
            state = clone_tree(self.model.state_dict(), cpu=True)
            backbone = getattr(self.model, "backbone", None)
            by_id = {id(p): n for n, p in self.model.named_parameters()}
            payload = {
                "schema_version": 1, "reason": reason, "step": self.step,
                "fit_config": asdict(self.config),
                "rank": self.rank, "world_size": self.world, "fit_seed": self.seed,
                "model_state": state, "training_modes": self.start_modes,
                "packing_spec": self.model.packing_spec.to_dict() if hasattr(self.model, "packing_spec") else None,
                "backbone_state": clone_tree(backbone.state_dict(), cpu=True) if backbone is not None else None,
                "pre_update_parameters": clone_tree(dict(self.model.named_parameters()) if before_update else self.pre_update["parameters"], cpu=True),
                "pre_forward_buffers": clone_tree(self.start_buffers, cpu=True),
                "optimizer_state": clone_tree(self.optimizer.state_dict() if before_update else self.pre_update["optimizer"], cpu=True),
                "optimizer_parameter_names": [[by_id[id(p)] for p in g["params"]] for g in self.optimizer.param_groups],
                "batch": clone_tree(self.batch, cpu=True), "rng": self.start_rng,
                "microbatch_rows": self.microbatch_rows, "gradient_clip_norm": self.config.gradient_clip_norm,
                "probe": clone_tree(self.probe, cpu=True), "metrics": self.metrics,
                "local_gradient_norm": getattr(self, "local_norm", None), "local_loss": getattr(self, "local_loss", None),
                "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            }
            directory = Path(self.config.diagnostic_snapshot_dir)
            directory.mkdir(parents=True, exist_ok=True)
            destination = Path(tempfile.mkdtemp(prefix=f"seed{self.seed}-step{self.step}-rank{self.rank}-", dir=directory))
            torch.save(payload, destination / "snapshot.pt.tmp")
            (destination / "snapshot.pt.tmp").replace(destination / "snapshot.pt")
            self.saved += 1
            LOG.warning("[classifier/stability] saved %s snapshot: %s", reason, destination / "snapshot.pt")
        except Exception:
            LOG.exception("[classifier/stability] snapshot failed on rank %s", self.rank)


def replay_snapshot(model, payload):
    """Replay one training update on an already reconstructed classifier.

For distributed captures, start the original world size and pass each rank's
own artifact. This intentionally mutates this disposable model and RNG.
"""
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported classifier diagnostic snapshot schema")
    world = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    rank = torch.distributed.get_rank() if world > 1 else 0
    if (world, rank) != (payload["world_size"], payload["rank"]):
        raise ValueError("Replay requires the captured world size and matching per-rank artifact")
    device = next(model.parameters()).device
    model.load_state_dict(payload["model_state"], strict=True)
    if payload["backbone_state"] is not None:
        model.backbone.load_state_dict(payload["backbone_state"], strict=True)
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in payload["pre_update_parameters"].items():
            parameters[name].copy_(value)
        for name, buffer in diagnostic_buffers(model).items():
            buffer.copy_(payload["pre_forward_buffers"][name])
    enabled = set(n for group in payload["optimizer_parameter_names"] for n in group)
    for name, parameter in parameters.items():
        parameter.requires_grad_(name in enabled)
    optimizer = torch.optim.AdamW([{"params": [parameters[n] for n in names]}
                                  for names in payload["optimizer_parameter_names"]])
    optimizer.load_state_dict(payload["optimizer_state"])
    for name, module in diagnostic_modules(model).items():
        module.training = payload["training_modes"][name]
    torch.set_float32_matmul_precision(payload["float32_matmul_precision"])
    torch.backends.cuda.matmul.allow_tf32 = payload["cuda_matmul_allow_tf32"]
    restore_rng(payload["rng"], device)
    batch = [t.to(device) for t in payload["batch"]]
    optimizer.zero_grad(set_to_none=True)
    total, size = 0., len(batch[0])
    for start in range(0, size, payload["microbatch_rows"]):
        stop = min(size, start + payload["microbatch_rows"])
        for offset in (0, 3):
            c, z, w = (t[start:stop] for t in batch[offset:offset+3])
            logits = model(c, z)
            loss = .5 * (stop-start)/size * (w * F.softplus(-logits if offset == 0 else logits)).mean()
            loss.backward()
            total += float(loss.detach())
    from .ratio_fit import _average_gradients
    _average_gradients(model)
    if payload["gradient_clip_norm"] is not None:
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], payload["gradient_clip_norm"])
    optimizer.step()
    return total
