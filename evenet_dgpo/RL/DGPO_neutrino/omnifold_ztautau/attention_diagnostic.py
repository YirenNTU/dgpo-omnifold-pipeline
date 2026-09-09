"""Opt-in, rank-local capture and replay of a failing classifier attention op.

No optimizer updates, backend changes, or distributed collectives are performed
by capture. Artifacts contain event data and must stay in the user's storage.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
import tempfile

import torch
from torch import nn

_log = logging.getLogger(__name__)
ENV_KEYS = ("DGPO_ATTN_DIAGNOSTIC_DIR", "DGPO_ATTN_DIAGNOSTIC_STEPS", "DGPO_ATTN_DIAGNOSTIC_SAVE_FIRST")
_saved_healthy_baseline = False


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_copy(v) for v in value)
    return value


def tensor_summary(tensor):
    if tensor is None:
        return None
    value = tensor.detach()
    finite = torch.isfinite(value)
    good = value[finite].float()
    return {
        "shape": list(value.shape), "dtype": str(value.dtype),
        "nonfinite": int((~finite).sum().item()),
        "min": float(good.min().item()) if good.numel() else None,
        "max": float(good.max().item()) if good.numel() else None,
    }


class AttentionFailureCapture:
    def __init__(self, model, condition, sample, weight, metadata):
        self.model = model
        self.batch = (condition, sample, weight)
        self.metadata = metadata
        self.directory = os.environ.get(ENV_KEYS[0], "")
        limit = int(os.environ.get(ENV_KEYS[1], "10"))
        self.enabled = bool(self.directory and int(metadata.get("step", 1)) <= limit)
        self.handles = []
        self.records = []

    def __enter__(self):
        if not self.enabled:
            return self
        self.batch_rng = torch.get_rng_state()
        self.device = self.batch[0].device
        self.batch_cuda_rng = (
            torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None
        )
        # Only hook the classifier decoder, not every PET block: retaining all
        # upstream attention graphs can materially increase debugging memory.
        roots = [("classifier", self.model)]
        seen = set()
        for prefix, root in roots:
            for name, module in root.named_modules():
                if not isinstance(module, nn.MultiheadAttention) or id(module) in seen:
                    continue
                seen.add(id(module))
                label = f"{prefix}.{name}"
                self.handles.append(module.register_forward_pre_hook(
                    lambda mod, args, kwargs, label=label: self._before(label, mod, args, kwargs),
                    with_kwargs=True,
                ))
                self.handles.append(module.register_forward_hook(self._after, with_kwargs=True))
        return self

    def _before(self, name, module, args, kwargs):
        kwargs = dict(kwargs)
        values = list(args)
        for key in ("query", "key", "value")[len(values):]:
            values.append(kwargs.pop(key))
        aliases = [next(i for i, other in enumerate(values) if other is value) for value in values[:3]]
        self.records.append({
            "name": name, "module": module, "args": tuple(values), "kwargs": kwargs,
            "input_aliases": aliases,
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None,
            "grad_output": None,
        })

    def _after(self, module, args, kwargs, output):
        record = next(row for row in reversed(self.records) if row["module"] is module)
        record["output"] = output[0].detach()
        if output[0].requires_grad:
            def save_gradient(gradient):
                record["grad_output"] = gradient.detach().clone()
                # Returning None leaves the gradient unchanged.
            self.handles.append(output[0].register_hook(save_gradient))

    def _save(self, error):
        directory = Path(self.directory).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        rank = self.metadata.get("rank", 0)
        target = Path(tempfile.mkdtemp(prefix=f"rank{rank}-step{self.metadata.get('step', 0)}-", dir=directory))
        records = []
        for row in self.records:
            module = row["module"]
            record = {k: cpu_copy(v) for k, v in row.items() if k != "module"}
            record["config"] = {
                "embed_dim": module.embed_dim, "num_heads": module.num_heads,
                "dropout": module.dropout, "bias": module.in_proj_bias is not None,
                "add_bias_kv": module.bias_k is not None, "add_zero_attn": module.add_zero_attn,
                "kdim": module.kdim, "vdim": module.vdim, "batch_first": module.batch_first,
            }
            record["state_dict"] = cpu_copy(module.state_dict())
            record["training"] = module.training
            record["requires_grad"] = {k: p.requires_grad for k, p in module.named_parameters()}
            record["input_requires_grad"] = [t.requires_grad for t in row["args"][:3]]
            records.append(record)
        backbone = getattr(self.model, "backbone", None)
        payload = {
            "schema_version": 1, "metadata": self.metadata, "error": str(error),
            "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
            "device": str(self.device),
            "device_name": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "cpu",
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "batch": cpu_copy(self.batch), "cpu_rng": self.batch_rng,
            "cuda_rng": self.batch_cuda_rng,
            "classifier_state_dict": cpu_copy(self.model.state_dict()),
            "backbone_state_dict": cpu_copy(backbone.state_dict()) if backbone is not None else None,
            "packing_spec": self.model.packing_spec.to_dict() if hasattr(self.model, "packing_spec") else None,
            "attention": records,
        }
        torch.save(payload, target / "failure.pt.tmp")
        os.replace(target / "failure.pt.tmp", target / "failure.pt")
        _log.error("[DGPO/attention-diagnostic] saved %s", target / "failure.pt")

    def __exit__(self, error_type, error, traceback):
        global _saved_healthy_baseline
        try:
            baseline = (
                os.environ.get(ENV_KEYS[2], "0") == "1"
                and not _saved_healthy_baseline
                and self.metadata.get("step") == 1 and self.metadata.get("rank") == 0
                and self.metadata.get("class") == "positive"
                and self.metadata.get("micro_start", 0) == 0
            )
            if self.enabled and (error is not None or baseline):
                self._save(error if error is not None else "healthy first-step baseline")
                if error is None:
                    _saved_healthy_baseline = True
        except Exception:
            # Never hide the original training failure or strand other ranks.
            _log.exception("[DGPO/attention-diagnostic] artifact save failed")
        finally:
            for handle in self.handles:
                handle.remove()
            self.records.clear()
        return False


def replay_attention(payload, *, device="cuda:0", backends=("efficient", "math"), dropout_zero=False):
    """Replay exact captured MHA inputs/weights/upstream gradients; never fit.

    Each backend gets the same RNG state, but backend-specific dropout masks
    need not match. dropout_zero is an explicitly labelled secondary test.
    """
    from torch.nn.attention import SDPBackend, sdpa_kernel
    device = torch.device(device)
    choices = {"efficient": SDPBackend.EFFICIENT_ATTENTION, "math": SDPBackend.MATH}
    results = []
    old_precision = torch.get_float32_matmul_precision()
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.set_float32_matmul_precision(payload["float32_matmul_precision"])
        torch.backends.cuda.matmul.allow_tf32 = payload["cuda_matmul_allow_tf32"]
        for record in payload["attention"]:
            if record["grad_output"] is None:
                results.append({"name": record["name"], "status": "not_reached_by_backward"})
                continue
            for backend in backends:
                row = {"name": record["name"], "backend": backend, "dropout_zero": dropout_zero}
                results.append(row)
                try:
                    with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
                        config = dict(record["config"])
                        if dropout_zero:
                            config["dropout"] = 0.0
                        dtype = record["args"][0].dtype
                        module = nn.MultiheadAttention(**config).to(device=device, dtype=dtype)
                        module.load_state_dict(record["state_dict"], strict=True)
                        module.train(record["training"])
                        for name, parameter in module.named_parameters():
                            parameter.requires_grad_(record["requires_grad"][name])
                        # Preserve aliasing for self-attention dispatch.
                        memo = {}
                        args = []
                        for index, value in enumerate(record["args"]):
                            if isinstance(value, torch.Tensor):
                                alias = record["input_aliases"][index] if index < 3 else index
                                if alias not in memo:
                                    memo[alias] = value.to(device).detach().clone().requires_grad_(
                                        record["input_requires_grad"][index] if index < 3 else False)
                                value = memo[alias]
                            args.append(value)
                        kwargs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in record["kwargs"].items()}
                        torch.set_rng_state(record["cpu_rng"])
                        if device.type == "cuda" and record["cuda_rng"] is not None:
                            torch.cuda.set_rng_state(record["cuda_rng"], device)
                        row["inputs"] = [tensor_summary(v) for v in args[:3]]
                        row["upstream_gradient"] = tensor_summary(record["grad_output"])
                        if module.in_proj_weight is not None:
                            weights = module.in_proj_weight.chunk(3)
                        else:
                            weights = (module.q_proj_weight, module.k_proj_weight, module.v_proj_weight)
                        biases = module.in_proj_bias.chunk(3) if module.in_proj_bias is not None else (None,) * 3
                        with torch.no_grad():
                            row["projected_qkv"] = [tensor_summary(torch.nn.functional.linear(v, w, b)) for v, w, b in zip(args[:3], weights, biases)]
                        mask = kwargs.get("key_padding_mask")
                        row["padding_mask"] = tensor_summary(mask)
                        row["attention_mask"] = tensor_summary(kwargs.get("attn_mask"))
                        if mask is not None:
                            blocked = mask if mask.dtype == torch.bool else torch.isneginf(mask)
                            row["fully_masked_rows"] = int(blocked.all(dim=-1).sum().item())
                        with sdpa_kernel(choices[backend]):
                            output = module(*args, **kwargs)[0]
                            row["output"] = tensor_summary(output)
                            row["captured_output"] = tensor_summary(record["output"])
                            row["absolute_output_difference_from_capture"] = tensor_summary(
                                (output.detach().cpu() - record["output"]).abs())
                            output.backward(record["grad_output"].to(device))
                        row["input_gradients"] = [tensor_summary(v.grad) for v in args[:3]]
                        row["parameter_gradients"] = {k: tensor_summary(v.grad) for k, v in module.named_parameters()}
                        summaries = [row["output"], row["upstream_gradient"], *row["inputs"], *row["projected_qkv"], *row["input_gradients"], *row["parameter_gradients"].values()]
                        row["status"] = "nonfinite" if any(v and v["nonfinite"] for v in summaries) else "finite"
                except Exception as error:
                    row.update(status="error", error=str(error))
    finally:
        torch.set_float32_matmul_precision(old_precision)
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    return {"capture": payload["metadata"], "capture_error": payload["error"],
            "batch": {k: tensor_summary(v) for k, v in zip(("condition", "sample", "weight"), payload["batch"])},
            "capture_environment": {k: payload[k] for k in ("torch_version", "cuda_version", "device_name")},
            "replay_environment": {"torch_version": str(torch.__version__), "device": str(device)},
            "dropout_note": "Same RNG state does not guarantee identical dropout masks across backends.",
            "results": results}
