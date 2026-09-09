"""Read captured classifier states and trace activation scales without fitting."""
from __future__ import annotations

import inspect
import math

import torch
from torch import nn

if __package__:
    from .attention_diagnostic import tensor_summary
else:  # Lightweight file-based CLI import avoids optional training packages.
    from attention_diagnostic import tensor_summary


def _valid_mask(value, mask):
    mask = mask.bool()
    while mask.ndim > value.ndim and mask.shape[-1] == 1:
        mask = mask.squeeze(-1)
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    return mask.expand_as(value)


def feature_summary(value, mask=None):
    """Separate padded values from valid features; report numeric feature indices."""
    rows = []
    valid = _valid_mask(value, mask) if mask is not None else torch.ones_like(value, dtype=torch.bool)
    for index in range(value.shape[-1]):
        column = value[..., index]
        selected = valid[..., index]
        row = {"feature_index": index, "valid": tensor_summary(column[selected]),
               "padding": tensor_summary(column[~selected])}
        finite = selected & torch.isfinite(column)
        if finite.any():
            magnitude = torch.where(finite, column.abs(), -torch.ones_like(column))
            offset = int(magnitude.reshape(-1).argmax())
            coordinates = []
            for width in reversed(column.shape):
                coordinates.append(offset % width)
                offset //= width
            row["max_abs_valid_location"] = list(reversed(coordinates)) + [index]
        rows.append(row)
    return rows


def inspect_artifact(payload):
    packed, sample, weight = payload["batch"]
    shapes = (payload.get("packing_spec") or {}).get("shapes", {})
    fields, offset = {}, 0
    for name, shape in shapes.items():
        width = math.prod(shape)
        fields[name] = packed[:, offset:offset + width].reshape(len(packed), *shape)
        offset += width
    if shapes and offset != packed.shape[-1]:
        raise ValueError("Captured packing spec does not match batch width")
    summaries = {}
    for name, value in fields.items():
        mask = fields.get({"x": "x_mask", "conditions": "conditions_mask"}.get(name, ""))
        summaries[name] = {"all": tensor_summary(value)}
        if name in ("x", "conditions"):
            summaries[name]["features"] = feature_summary(value, mask)
    state = payload.get("backbone_state_dict") or {}
    normalizers = {}
    for key, value in state.items():
        if "normalizer" in key and key.rsplit(".", 1)[-1] in ("mean", "std", "norm_mask"):
            normalizers[key] = {"summary": tensor_summary(value), "values": [
                float(v) if math.isfinite(float(v)) else str(float(v)) for v in value.reshape(-1)]}
    attention = []
    for row in payload["attention"]:
        mask = row["kwargs"].get("key_padding_mask")
        memory = row["args"][1]
        if mask is not None and not row["config"]["batch_first"]:
            memory = memory.transpose(0, 1)
        valid = None if mask is None else ~(mask if mask.dtype == torch.bool else torch.isneginf(mask))
        attention.append({"name": row["name"], "query": tensor_summary(row["args"][0]),
                          "memory": tensor_summary(memory),
                          "valid_memory": tensor_summary(memory[_valid_mask(memory, valid)]) if valid is not None else None,
                          "padded_memory": tensor_summary(memory[~_valid_mask(memory, valid)]) if valid is not None else None,
                          "output": tensor_summary(row.get("output"))})
    return {"capture": payload["metadata"], "error": payload["error"], "packed_fields": summaries,
            "sample": tensor_summary(sample), "weight": tensor_summary(weight),
            "normalizers": normalizers, "attention": attention}


def compare_states(left, right):
    changed = []
    structural = []
    for section in ("backbone_state_dict", "classifier_state_dict"):
        a, b = left.get(section) or {}, right.get(section) or {}
        for key in sorted(a.keys() | b.keys()):
            label = f"{section}.{key}"
            if key not in a or key not in b or a[key].shape != b[key].shape:
                structural.append(label)
                continue
            if torch.equal(a[key], b[key]):
                continue
            changed.append({"key": label, "absolute_difference": tensor_summary((a[key].double() - b[key].double()).abs()),
                            "left": tensor_summary(a[key]), "right": tensor_summary(b[key])})
    changed.sort(key=lambda row: row["absolute_difference"]["max"] or 0, reverse=True)
    return {"changed_tensor_count": len(changed), "structural_differences": structural,
            "largest_changes": changed[:25],
            "normalizer_changes": [row for row in changed if "normalizer" in row["key"]],
            "note": "Different batches confound before/after scales. Trace with --weights-from to hold the batch fixed."}


def trace_forward(model, batch_payload, *, device="cuda:0", backend="original", compare_capture=True):
    """Trace one forward, retaining training/autograd mode; no backward or optimizer."""
    from contextlib import nullcontext
    from torch.nn.attention import SDPBackend, sdpa_kernel
    device = torch.device(device)
    stages, handles = [], []
    originals = {row["name"]: row for row in batch_payload["attention"]}
    def summary_tree(value):
        if isinstance(value, torch.Tensor):
            return tensor_summary(value)
        if isinstance(value, (list, tuple)):
            return [summary_tree(v) for v in value]
        if isinstance(value, dict):
            return {k: summary_tree(v) for k, v in value.items()}
        return None
    def before(name, module, args, kwargs):
        row = {"stage": name, "input_args": summary_tree(args), "input_kwargs": summary_tree(kwargs)}
        if isinstance(module, nn.MultiheadAttention) and compare_capture and name in originals:
            original = originals[name]
            values = list(args[:3])
            for key in ("query", "key", "value")[len(values):]:
                values.append(kwargs[key])
            row["absolute_input_difference_from_capture"] = [
                tensor_summary((value.detach().cpu() - old).abs())
                for value, old in zip(values, original["args"][:3])]
        stages.append(row)
    def after(name, module, args, kwargs, output):
        row = next(row for row in reversed(stages) if row["stage"] == name and "output" not in row)
        row["output"] = summary_tree(output)
    backbone = getattr(model, "backbone", None)
    roots = [("classifier", model)] + ([("backbone", backbone)] if backbone is not None else [])
    seen = set()
    for prefix, root in roots:
        for name, module in root.named_modules():
            selected = prefix == "classifier" and (isinstance(module, nn.MultiheadAttention) or name in ("bank.decoder.candidate_in", "bank.decoder.memory_in"))
            selected |= prefix == "backbone" and (name in ("sequential_normalizer", "global_normalizer", "invisible_normalizer", "GroupedSequentialEmbedding", "InvisibleInputProjector", "GlobalEmbedding", "PET") or name.startswith("PET.") and module.__class__.__name__ in ("TransformerBlockModule", "Adapter"))
            if not selected or id(module) in seen:
                continue
            seen.add(id(module))
            label = f"{prefix}.{name}"
            handles.append(module.register_forward_pre_hook(lambda mod, a, k, label=label: before(label, mod, a, k), with_kwargs=True))
            handles.append(module.register_forward_hook(lambda mod, a, k, o, label=label: after(label, mod, a, k, o), with_kwargs=True))
    old_precision = torch.get_float32_matmul_precision()
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    result = {"stages": stages, "backend": backend, "capture": batch_payload["metadata"]}
    try:
        torch.set_float32_matmul_precision(batch_payload["float32_matmul_precision"])
        torch.backends.cuda.matmul.allow_tf32 = batch_payload["cuda_matmul_allow_tf32"]
        with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
            condition, sample, _ = batch_payload["batch"]
            condition, sample = condition.to(device), sample.to(device)
            torch.set_rng_state(batch_payload["cpu_rng"])
            if device.type == "cuda" and batch_payload["cuda_rng"] is not None:
                torch.cuda.set_rng_state(batch_payload["cuda_rng"], device)
            context = nullcontext() if backend == "original" else sdpa_kernel(SDPBackend.MATH)
            # no_grad would change frozen PET's attention dispatch, defeating
            # comparison with the captured classifier training forward.
            with context, torch.enable_grad():
                output = model(condition, sample)
                result["logits"] = tensor_summary(output)
            result["status"] = "nonfinite" if result["logits"]["nonfinite"] else "finite_forward"
    except Exception as error:
        result.update(status="error", error=str(error))
    finally:
        for handle in handles:
            handle.remove()
        torch.set_float32_matmul_precision(old_precision)
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    return result


def trace_artifact(payload, runtime, *, device="cuda:0", backend="original", weights_payload=None):
    import yaml
    from RL.DGPO_neutrino.model_utils import load_training_config, load_normalization_dict
    from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import EvenetAdapterModelBuilder, EventPackingSpec
    with open(runtime) as stream:
        raw = yaml.safe_load(stream)
    source = payload if weights_payload is None else weights_payload
    if payload["packing_spec"] != source["packing_spec"]:
        raise ValueError("Cross-state trace requires identical packing protocols")
    if source.get("backbone_state_dict") is None:
        raise ValueError("Trace requires a captured classifier backbone")
    config = load_training_config(runtime)
    recalibration = raw["dgpo"]["adaptive_omnifold"]["recalibration"]
    supported = inspect.signature(EvenetAdapterModelBuilder).parameters
    kwargs = {k: v for k, v in recalibration.items() if k in supported}
    builder = EvenetAdapterModelBuilder(
        config=config, normalization_dict=load_normalization_dict(config),
        checkpoint_path=raw["reward_config"]["omnifold"]["backbone_checkpoint"],
        device=torch.device(device), **kwargs)
    model = builder.make_classifier(EventPackingSpec.from_dict(payload["packing_spec"]), reset=True).to(device)
    model.backbone.load_state_dict(source["backbone_state_dict"], strict=True)
    model.load_state_dict(source["classifier_state_dict"], strict=True)
    model.train(True)
    result = trace_forward(model, payload, device=device, backend=backend, compare_capture=weights_payload is None)
    result["weight_capture"] = source["metadata"]
    result["raw_sequential_feature_names"] = model.backbone._raw_sequential_feature_names()
    event_info = model.backbone.event_info
    result["global_feature_names"] = [feature.name for name, kind in event_info.input_types.items()
                                      if str(kind).upper() == "GLOBAL"
                                      for feature in event_info.input_features[name]]
    result["note"] = "Diagnostic forward only. Earlier weights with the same failed batch separate batch effects from parameter updates; no fitting occurs."
    return result
