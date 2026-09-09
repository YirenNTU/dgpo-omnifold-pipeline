#!/usr/bin/env python3
"""Compare effective 1%/10% configs or replay captured classifier attention."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evenet_dgpo"))
sys.path.insert(0, str(ROOT / "scripts"))


def compare_configs(base, one, ten):
    from train_neutrino_backend import read_yaml, deep_update, absolutize_default_paths
    def effective(path):
        return absolutize_default_paths(deep_update(read_yaml(base), read_yaml(path)), base.parent)
    left, right = effective(one), effective(ten)
    differences = []
    def walk(a, b, prefix=""):
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(a.keys() | b.keys()):
                walk(a.get(key), b.get(key), f"{prefix}.{key}" if prefix else key)
        elif a != b:
            differences.append({"key": prefix, "one_pct": a, "ten_pct": b})
    walk(left, right)
    return {"one_pct": str(one), "ten_pct": str(ten), "differences": differences,
            "note": "Base+overlay YAML comparison; checkpoint contents and generated batches are not inspected."}


def prepare_capture(fraction, output_dir, checkpoint=None):
    """Cold-start a control in new storage; never touch production checkpoints."""
    import yaml
    from train_neutrino_backend import read_yaml, deep_update, absolutize_default_paths
    base = ROOT / "config/train_diffusion_nersc.yaml"
    overlay = ROOT / f"config/dgpo_omnifold_ztautau_{fraction}pct_scaling_raw_plateau_vpkl.yaml"
    config = absolutize_default_paths(deep_update(read_yaml(base), read_yaml(overlay)), base.parent)
    checkpoint = Path(checkpoint or config["options"]["Training"]["model_checkpoint_load_path"]).expanduser().resolve(strict=True)
    with checkpoint.open("rb") as stream:
        checkpoint_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    training = config["options"]["Training"]
    training["model_checkpoint_load_path"] = str(checkpoint)
    training["model_checkpoint_save_path"] = str(output_dir / "checkpoints")
    config["reward_config"]["omnifold"]["backbone_checkpoint"] = str(checkpoint)
    config["dgpo"]["auto_resume_from_last"] = False
    config["dgpo"]["auto_resume_best_source_checkpoint_dir"] = None
    config["dgpo"]["auto_resume_fallback_checkpoint_path"] = None
    config["dgpo"]["checkpoint_load_mode"] = "weights_only"
    config["dgpo"]["adaptive_omnifold"]["recalibration"]["refit_once_on_resume"] = False
    # Same debug mode for both fractions. No LR, mask, dropout, loss, data,
    # batch-size or closure changes; use v18 on BOTH sides of the comparison.
    name = f"omnifold_attention_{fraction}pct_{output_dir.name}"
    config["logger"]["wandb"].update(run_name=name, resume="never")
    config["logger"]["local"].update(save_dir=str(output_dir / "logs"), name=name, version="attention_diagnostic")
    config["nersc"]["ray"]["results_dir"] = str(output_dir / "ray_results")
    config["nersc"]["execution"].pop("command", None)
    config["attention_diagnostic"] = {"checkpoint_sha256": checkpoint_sha256, "source_overlay": str(overlay)}
    runtime = output_dir / "runtime.yaml"
    with runtime.open("x") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    return runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    compare = commands.add_parser("compare")
    compare.add_argument("--base", type=Path, default=ROOT / "config/train_diffusion_nersc.yaml")
    compare.add_argument("--one", type=Path, default=ROOT / "config/dgpo_omnifold_ztautau_1pct_scaling_raw_plateau_vpkl.yaml")
    compare.add_argument("--ten", type=Path, default=ROOT / "config/dgpo_omnifold_ztautau_10pct_scaling_raw_plateau_vpkl.yaml")
    capture = commands.add_parser("capture", help="Launch a fresh, isolated v18 classifier diagnostic on Ray")
    capture.add_argument("--fraction", choices=("1", "10"), required=True)
    capture.add_argument("--output-dir", type=Path, required=True, help="New shared-storage directory; must not exist")
    capture.add_argument("--checkpoint", type=Path, help="Optional alternate pretrained checkpoint; used for both policy and classifier")
    capture.add_argument("--prepare-only", action="store_true", help="Write isolated runtime config without launching Ray")
    replay = commands.add_parser("replay")
    replay.add_argument("artifact", type=Path, help="Trusted failure.pt captured by this diagnostic")
    replay.add_argument("--device", default="cuda:0")
    replay.add_argument("--backends", nargs="+", choices=("efficient", "math"), default=["efficient", "math"])
    replay.add_argument("--dropout-zero", action="store_true", help="Secondary test only; intentionally changes attention dropout")
    replay.add_argument("--output", type=Path, help="New JSON report file; refuses overwrite")
    inspect_command = commands.add_parser("inspect", help="Inspect packed feature scales, padding, normalizers, and saved weights")
    inspect_command.add_argument("artifact", type=Path)
    inspect_command.add_argument("--compare", type=Path, help="Second capture, e.g. the healthy step-1 artifact")
    inspect_command.add_argument("--output", type=Path, help="New JSON report; refuses overwrite")
    trace = commands.add_parser("trace", help="Trace one classifier forward, without retraining or updating weights")
    trace.add_argument("artifact", type=Path)
    trace.add_argument("--runtime", type=Path, required=True, help="The capture run's runtime.yaml")
    trace.add_argument("--weights-from", type=Path, help="Use another capture's weights with this artifact's batch and RNG")
    trace.add_argument("--device", default="cuda:0")
    trace.add_argument("--backend", choices=("original", "math"), default="original")
    trace.add_argument("--output", type=Path, help="New JSON report; refuses overwrite")
    trace.add_argument("--summary-only", action="store_true", help="Print compact stage scales; --output still saves full trace")
    args = parser.parse_args()
    if args.command == "capture":
        runtime = prepare_capture(args.fraction, args.output_dir, args.checkpoint)
        print(f"Diagnostic runtime: {runtime}", flush=True)
        if args.prepare_only:
            return 0
        env = dict(os.environ)
        env.update(DGPO_ATTN_DIAGNOSTIC_DIR=str(runtime.parent / "attention_artifacts"),
                   DGPO_ATTN_DIAGNOSTIC_STEPS="10", DGPO_ATTN_DIAGNOSTIC_SAVE_FIRST="1")
        env["PYTHONPATH"] = str(ROOT / "evenet_dgpo") + os.pathsep + env.get("PYTHONPATH", "")
        # --max-steps bounds policy updates, not bootstrap classifier steps.
        command = [sys.executable, str(ROOT / "evenet_dgpo/RL/DGPO_neutrino/dgpo_trainer.py"),
                   str(runtime), "--no-wandb", "--max-steps", "1",
                   "--ray-dir", str(runtime.parent / "ray_results")]
        return subprocess.run(command, cwd=ROOT, env=env, check=False).returncode
    if args.command == "compare":
        result = compare_configs(args.base, args.one, args.ten)
    else:
        import torch
        # The replay only needs PyTorch; avoid importing the training package's
        # Lightning/Ray/torchvision dependencies through its __init__.
        import importlib.util
        spec = importlib.util.spec_from_file_location("attention_diagnostic", ROOT / "evenet_dgpo/RL/DGPO_neutrino/omnifold_ztautau/attention_diagnostic.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        payload = torch.load(args.artifact, map_location="cpu", weights_only=True)
        if payload.get("schema_version") != 1:
            parser.error("Unsupported diagnostic artifact schema")
        if args.command == "replay":
            result = module.replay_attention(payload, device=args.device, backends=args.backends, dropout_zero=args.dropout_zero)
        else:
            spec = importlib.util.spec_from_file_location("activation_diagnostic", ROOT / "evenet_dgpo/RL/DGPO_neutrino/omnifold_ztautau/activation_diagnostic.py")
            activation = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(activation)
            if args.command == "inspect":
                result = {"primary": activation.inspect_artifact(payload)}
                if args.compare:
                    other = torch.load(args.compare, map_location="cpu", weights_only=True)
                    result["comparison"] = activation.inspect_artifact(other)
                    result["weight_changes"] = activation.compare_states(other, payload)
            else:
                weights = torch.load(args.weights_from, map_location="cpu", weights_only=True) if args.weights_from else None
                result = activation.trace_artifact(payload, args.runtime, device=args.device, backend=args.backend, weights_payload=weights)
    rendered = json.dumps(result, indent=2, allow_nan=False)
    if args.command in {"replay", "inspect", "trace"} and args.output:
        with args.output.open("x") as stream:
            stream.write(rendered + "\n")
    if args.command == "trace" and args.summary_only:
        def ranges(value):
            if isinstance(value, dict):
                if "nonfinite" in value and "min" in value and "max" in value:
                    return [value]
                return [item for child in value.values() for item in ranges(child)]
            if isinstance(value, list):
                return [item for child in value for item in ranges(child)]
            return []
        print("status:", result["status"], "error:", result.get("error"))
        print("weight_capture:", result.get("weight_capture"))
        for row in result["stages"]:
            outputs = ranges(row.get("output"))
            differences = ranges(row.get("absolute_input_difference_from_capture"))
            extremes = [abs(v[k]) for v in outputs for k in ("min", "max") if v[k] is not None]
            delta = [v["max"] for v in differences if v["max"] is not None]
            print(row["stage"], "output_maxabs=", max(extremes, default=None),
                  "nonfinite=", sum(v["nonfinite"] for v in outputs),
                  "capture_input_maxdiff=", max(delta, default=None))
        if args.output:
            print("Full trace:", args.output)
    else:
        print(rendered)
    if args.command == "replay" and any(row["status"] in {"error", "nonfinite"} for row in result["results"]):
        return 1
    if args.command == "trace" and result["status"] in {"error", "nonfinite"}:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
