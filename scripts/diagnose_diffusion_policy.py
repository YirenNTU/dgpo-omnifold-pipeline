#!/usr/bin/env python3
"""Test live pretrained DDIM inference on complete captured classifier batches.

No classifier fit, policy update, EMA substitution, or input sanitization.
This is a numerical inference test, not a physics-quality or backward test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evenet_dgpo"))


def summary(value):
    finite = torch.isfinite(value)
    good = value.detach()[finite].float()
    return {"shape": list(value.shape), "nonfinite": int((~finite).sum()),
            "min": float(good.min()) if good.numel() else None,
            "max": float(good.max()) if good.numel() else None}


class CheckedPolicy:
    def __init__(self, model):
        self.model = model
        self.invisible_input_dim = model.invisible_input_dim
        self.invisible_normalizer = model.invisible_normalizer
        self.calls = []

    def predict_diffusion_vector(self, *args, **kwargs):
        output = self.model.predict_diffusion_vector(*args, **kwargs)
        stats = summary(output)
        self.calls.append(stats)
        if stats["nonfinite"]:
            raise FloatingPointError("Nonfinite diffusion velocity before sampler postprocessing")
        return output


def check_batches(model, fields, *, device, batch_size, ddim_steps, seed, generate, sampler):
    """Cover every captured row, including outliers beyond the first microbatch."""
    if batch_size < 1 or ddim_steps < 1:
        raise ValueError("batch size and DDIM steps must be positive")
    count = len(fields["x"])
    if count < 1 or any(len(value) != count for value in fields.values()):
        raise ValueError("Expected nonempty fields with matching event counts")
    rows = []
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        batch = {key: value[start:stop].to(device) for key, value in fields.items()}
        # Ztautau has two neutrino slots. Only shape/mask is consumed by DDIM;
        # no truth candidate or captured classifier weights are used as input.
        batch["x_invisible"] = torch.zeros(stop-start, 2, model.invisible_input_dim, device=device)
        batch["x_invisible_mask"] = torch.ones(stop-start, 2, dtype=torch.bool, device=device)
        checked = CheckedPolicy(model)
        result = {"start": start, "stop": stop, "velocity": checked.calls}
        try:
            torch.manual_seed(seed + start)
            with torch.no_grad():
                samples = generate(checked, batch, sampler, K=1, num_ddim_steps=ddim_steps,
                                   device=device, parallel_chains=1)
            result["generated"] = summary(samples)
            if not checked.calls:
                raise RuntimeError("Sampler made no diffusion velocity calls")
            result["status"] = "nonfinite" if result["generated"]["nonfinite"] else "finite"
        except Exception as exc:
            result.update(status="error", error=f"{type(exc).__name__}: {exc}")
        rows.append(result)
        print(f"rows {start}:{stop}: {result['status']}", flush=True)
    return {"events": count, "status": "finite" if all(r["status"] == "finite" for r in rows) else "failed",
            "batches": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--runtime", type=Path, required=True, help="Merged runtime.yaml from capture")
    parser.add_argument("--checkpoint", type=Path, help="Defaults to the runtime's pretrained policy checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="New JSON report; refuses overwrite")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ddim-steps", type=int, help="Defaults to runtime dgpo.num_ddim_steps")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Report already exists; choose a new --output")
    from RL.DGPO_neutrino.model_utils import load_training_config, load_evenet_model_for_dgpo
    from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import EventPackingSpec, unpack_event_inputs
    from RL.DGPO_neutrino.sampling import generate_neutrino_candidates
    from evenet.utilities.diffusion_sampler import DDIMSampler

    config = load_training_config(args.runtime)
    config.options.Training.EMA.replace_model_after_load = False
    config.options.Training.EMA.use_for_generation = False
    checkpoint = Path(args.checkpoint or config.options.Training.model_checkpoint_load_path).resolve(strict=True)
    steps = args.ddim_steps if args.ddim_steps is not None else int(config.dgpo.num_ddim_steps)
    if args.batch_size < 1 or steps < 1:
        parser.error("batch size and DDIM steps must be positive")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision(str(config.dgpo.get("float32_matmul_precision", "highest")))
    with checkpoint.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    model = load_evenet_model_for_dgpo(config=config, checkpoint_path=checkpoint, device=device).model
    model.eval()
    bad_weights = [name for name, value in model.state_dict().items() if not torch.isfinite(value).all()]
    report = {"checkpoint": str(checkpoint), "checkpoint_sha256": digest,
              "weight_source": "state_dict", "mode": "eval, no_grad, original attention dispatch",
              "torch": str(torch.__version__), "device": str(device),
              "batch_size": args.batch_size, "ddim_steps": steps, "seed": args.seed,
              "nonfinite_loaded_state_keys": bad_weights,
              "raw_sequential_feature_names": model._raw_sequential_feature_names(),
              "cases": [], "limitation": "Finite inference does not prove physics accuracy, stable backward, or correct parquet. Inputs are captured post-loader tensors."}
    sampler = DDIMSampler(device=device)
    for path in args.artifacts:
        print(f"Testing {path}", flush=True)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("schema_version") != 1:
            raise ValueError("Unsupported capture schema")
        if payload["batch"][1].ndim != 2 or payload["batch"][1].shape[1] != 4:
            raise ValueError("This diagnostic expects Ztautau captures with two neutrino angle pairs")
        fields = unpack_event_inputs(payload["batch"][0], EventPackingSpec.from_dict(payload["packing_spec"]))
        case = {"artifact": str(path), "capture": payload["metadata"],
                "inputs": {name: summary(value) for name, value in fields.items()}}
        case.update(check_batches(model, fields, device=device, batch_size=args.batch_size,
                    ddim_steps=steps, seed=args.seed, generate=generate_neutrino_candidates, sampler=sampler))
        report["cases"].append(case)
    report["status"] = "finite" if not bad_weights and all(c["status"] == "finite" for c in report["cases"]) else "failed"
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"{report['status']}: report saved to {args.output}")
    return 0 if report["status"] == "finite" else 1


if __name__ == "__main__":
    raise SystemExit(main())
