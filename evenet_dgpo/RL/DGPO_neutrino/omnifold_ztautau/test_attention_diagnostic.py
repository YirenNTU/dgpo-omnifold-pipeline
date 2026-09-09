"""CPU contracts for capture; real CUDA backend comparison runs on NERSC."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import importlib.util
import json
import contextlib
import io
import runpy
import sys

import torch
from torch import nn

from .attention_diagnostic import AttentionFailureCapture, replay_attention
from .activation_diagnostic import feature_summary, inspect_artifact, compare_states, trace_forward


class ToyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(8, 2, dropout=0.25, batch_first=True)

    def forward(self, x, sample=None):
        return self.cross_attn(query=x, key=x, value=x, need_weights=False)[0]


class TestAttentionDiagnostic(unittest.TestCase):
    def test_feature_summary_separates_padding_and_finds_large_field(self):
        values = torch.tensor([[[2., 1.e9], [float("nan"), 1.e12]]])
        mask = torch.tensor([[[True], [False]]])
        rows = feature_summary(values, mask)
        self.assertEqual(rows[0]["valid"]["nonfinite"], 0)
        self.assertEqual(rows[0]["padding"]["nonfinite"], 1)
        self.assertEqual(rows[1]["valid"]["max"], 1.e9)
        self.assertEqual(rows[1]["max_abs_valid_location"], [0, 0, 1])

    def test_inspect_unpack_and_normalizer_changes(self):
        payload = {"metadata": {"step": 1}, "error": "baseline", "attention": [],
                   "packing_spec": {"shapes": {"x": [1, 2], "x_mask": [1, 1]}},
                   "batch": (torch.tensor([[3., 1.e9, 1.]]), torch.zeros(1, 4), torch.ones(1)),
                   "backbone_state_dict": {"sequential_normalizer.std": torch.tensor([1., 1.e-6])}}
        result = inspect_artifact(payload)
        self.assertEqual(result["packed_fields"]["x"]["features"][1]["valid"]["max"], 1.e9)
        other = dict(payload, backbone_state_dict={"sequential_normalizer.std": torch.tensor([1., 1.])})
        diff = compare_states(payload, other)
        self.assertEqual(diff["changed_tensor_count"], 1)
        self.assertEqual(len(diff["normalizer_changes"]), 1)

    def test_successful_capture_does_not_change_forward_gradients_or_rng(self):
        model = ToyAttention()
        x = torch.randn(3, 4, 8)
        torch.manual_seed(4)
        output = model(x)
        output.square().sum().backward()
        expected = {k: p.grad.clone() for k, p in model.named_parameters()}
        rng = torch.get_rng_state().clone()
        model.zero_grad(set_to_none=True)
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            "DGPO_ATTN_DIAGNOSTIC_DIR": directory, "DGPO_ATTN_DIAGNOSTIC_SAVE_FIRST": "0",
        }):
            torch.manual_seed(4)
            with AttentionFailureCapture(model, x, x, x, {"step": 1}):
                observed = model(x)
                observed.square().sum().backward()
            torch.testing.assert_close(observed, output)
            torch.testing.assert_close(torch.get_rng_state(), rng)
            for name, parameter in model.named_parameters():
                torch.testing.assert_close(parameter.grad, expected[name])
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_prepare_capture_pins_checkpoint_and_refuses_existing_output(self):
        import yaml
        script = Path(__file__).resolve().parents[4] / "scripts/diagnose_omnifold_attention.py"
        spec = importlib.util.spec_from_file_location("diagnose_test", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "pretrain.ckpt"
            torch.save({"state_dict": {}}, checkpoint)
            output = Path(directory) / "diagnostic"
            runtime = module.prepare_capture("10", output, checkpoint)
            config = yaml.safe_load(runtime.read_text())
            self.assertFalse(config["dgpo"]["auto_resume_from_last"])
            self.assertEqual(config["options"]["Training"]["model_checkpoint_load_path"], str(checkpoint.resolve()))
            self.assertEqual(config["reward_config"]["omnifold"]["backbone_checkpoint"], str(checkpoint.resolve()))
            self.assertFalse(config["options"]["Training"]["EMA"]["replace_model_after_load"])
            self.assertEqual(config["dgpo"]["reference_trust"]["coefficient"], 1)
            self.assertEqual(len(config["attention_diagnostic"]["checkpoint_sha256"]), 64)
            with self.assertRaises(FileExistsError):
                module.prepare_capture("10", output, checkpoint)

    def test_disabled_capture_does_not_attach_hooks(self):
        model = ToyAttention()
        x = torch.randn(3, 4, 8)
        with patch.dict(os.environ, {"DGPO_ATTN_DIAGNOSTIC_DIR": ""}):
            with AttentionFailureCapture(model, x, x, x, {"step": 1}) as capture:
                self.assertFalse(capture.enabled)
                self.assertFalse(model.cross_attn._forward_hooks)

    def test_capture_replays_actual_gradient_and_preserves_rng_weights_and_aliases(self):
        torch.manual_seed(12)
        model = ToyAttention()
        x = torch.randn(3, 4, 8, requires_grad=True)
        initial = {k: v.clone() for k, v in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            "DGPO_ATTN_DIAGNOSTIC_DIR": directory, "DGPO_ATTN_DIAGNOSTIC_STEPS": "10",
        }):
            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                with AttentionFailureCapture(model, x, x, x, {"step": 1, "rank": 0}):
                    output = model(x)
                    output.square().sum().backward()
                    raise RuntimeError("synthetic failure")
            artifact = next(Path(directory).glob("*/failure.pt"))
            payload = torch.load(artifact, weights_only=True)
            self.assertEqual(payload["attention"][0]["input_aliases"], [0, 0, 0])
            self.assertFalse(model.cross_attn._forward_hooks)
            self.assertFalse(model.cross_attn._forward_pre_hooks)
            for k, value in model.state_dict().items():
                torch.testing.assert_close(value, initial[k])
            rng = torch.get_rng_state().clone()
            report = replay_attention(payload, device="cpu", backends=("math",))
            self.assertEqual(report["results"][0]["status"], "finite")
            torch.testing.assert_close(torch.get_rng_state(), rng)
            row = report["results"][0]
            captured = payload["attention"][0]
            self.assertAlmostEqual(row["output"]["max"], captured["output"].max().item())
            self.assertGreater(row["upstream_gradient"]["max"], 0)
            script = Path(__file__).resolve().parents[4] / "scripts/diagnose_omnifold_attention.py"
            stdout = io.StringIO()
            with patch.object(sys, "argv", [str(script), "replay", str(artifact), "--device", "cpu", "--backends", "math"]):
                with contextlib.redirect_stdout(stdout), self.assertRaises(SystemExit) as completed:
                    runpy.run_path(str(script), run_name="__main__")
            self.assertEqual(completed.exception.code, 0)
            self.assertEqual(json.loads(stdout.getvalue())["results"][0]["status"], "finite")
            before = {k: v.clone() for k, v in model.state_dict().items()}
            grads = {k: p.grad.clone() for k, p in model.named_parameters()}
            trace = trace_forward(model, payload, device="cpu")
            self.assertEqual(trace["status"], "finite_forward")
            torch.testing.assert_close(torch.get_rng_state(), rng)
            differences = trace["stages"][0]["absolute_input_difference_from_capture"]
            self.assertTrue(all(row["max"] == 0 for row in differences))
            for name, parameter in model.named_parameters():
                torch.testing.assert_close(parameter, before[name])
                torch.testing.assert_close(parameter.grad, grads[name])
            self.assertFalse(model.cross_attn._forward_hooks)
            stdout = io.StringIO()
            with patch.object(sys, "argv", [str(script), "inspect", str(artifact), "--compare", str(artifact)]):
                with contextlib.redirect_stdout(stdout), self.assertRaises(SystemExit) as completed:
                    runpy.run_path(str(script), run_name="__main__")
            self.assertEqual(completed.exception.code, 0)
            self.assertEqual(json.loads(stdout.getvalue())["weight_changes"]["changed_tensor_count"], 0)

    def test_save_failure_does_not_hide_original_exception(self):
        model = ToyAttention()
        x = torch.randn(3, 4, 8)
        with patch.dict(os.environ, {"DGPO_ATTN_DIAGNOSTIC_DIR": "unused"}):
            with patch.object(AttentionFailureCapture, "_save", side_effect=OSError("disk full")):
                with self.assertLogs(level="ERROR"), self.assertRaisesRegex(RuntimeError, "original"):
                    with AttentionFailureCapture(model, x, x, x, {"step": 1}):
                        raise RuntimeError("original")
        self.assertFalse(model.cross_attn._forward_hooks)

    def test_budget_disables_late_capture(self):
        x = torch.randn(1, 2, 8)
        with patch.dict(os.environ, {"DGPO_ATTN_DIAGNOSTIC_DIR": "unused", "DGPO_ATTN_DIAGNOSTIC_STEPS": "10"}):
            with AttentionFailureCapture(ToyAttention(), x, x, x, {"step": 11}) as capture:
                self.assertFalse(capture.enabled)


if __name__ == "__main__":
    unittest.main()
