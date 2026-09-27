"""Focused tests for reversible and RMS-scaled AdamW proposals."""

from __future__ import annotations

import copy
import unittest

import torch

from RL.DGPO_neutrino.optimizer_transaction import (
    assign_scaled_trainable_update_,
    resolve_parameter_update_rms_calibration,
)


class TestParameterUpdateRmsCalibration(unittest.TestCase):
    def test_config_validation(self):
        target, minimum, maximum = resolve_parameter_update_rms_calibration({
            "parameter_update_rms_calibration": {
                "enabled": True,
                "target_rms": 3.0e-5,
                "minimum_scale": 0.1,
                "maximum_scale": 100.0,
            }
        })
        self.assertEqual(target, 3.0e-5)
        self.assertEqual((minimum, maximum), (0.1, 100.0))
        self.assertEqual(
            resolve_parameter_update_rms_calibration({}),
            (None, 1.0, 1.0),
        )
        for payload in (
            {"enabled": True, "target_rms": 0.0},
            {
                "enabled": True,
                "target_rms": 3.0e-5,
                "minimum_scale": 2.0,
                "maximum_scale": 1.0,
            },
            {"enabled": True, "target_rms": 3.0e-5, "unknown": 1},
        ):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                resolve_parameter_update_rms_calibration({
                    "parameter_update_rms_calibration": payload
                })

    def test_scaled_proposal_equals_same_adamw_step_at_scaled_group_lr(self):
        first = torch.nn.Linear(2, 1, bias=True)
        second = copy.deepcopy(first)
        first_optimizer = torch.optim.AdamW(
            first.parameters(), lr=2.0e-4, betas=(0.8, 0.95),
            eps=1.0e-7, weight_decay=0.03,
        )
        second_optimizer = torch.optim.AdamW(
            second.parameters(), lr=2.0e-4, betas=(0.8, 0.95),
            eps=1.0e-7, weight_decay=0.03,
        )

        for model, optimizer in (
            (first, first_optimizer),
            (second, second_optimizer),
        ):
            optimizer.zero_grad(set_to_none=True)
            model.weight.grad = torch.tensor([[0.2, -0.4]])
            model.bias.grad = torch.tensor([0.1])
            optimizer.step()

        old = {
            name: parameter.detach().clone()
            for name, parameter in first.named_parameters()
        }
        scale = 7.0
        first_optimizer.zero_grad(set_to_none=True)
        first.weight.grad = torch.tensor([[-0.3, 0.6]])
        first.bias.grad = torch.tensor([-0.2])
        first_optimizer.step()
        candidate = {
            name: parameter.detach().clone()
            for name, parameter in first.named_parameters()
        }
        assign_scaled_trainable_update_(first, old, candidate, scale)

        for group in second_optimizer.param_groups:
            group["lr"] *= scale
        second_optimizer.zero_grad(set_to_none=True)
        second.weight.grad = torch.tensor([[-0.3, 0.6]])
        second.bias.grad = torch.tensor([-0.2])
        second_optimizer.step()

        for left, right in zip(first.parameters(), second.parameters(), strict=True):
            self.assertTrue(torch.allclose(left, right, rtol=2.0e-5, atol=2.0e-7))
            left_state = first_optimizer.state[left]
            right_state = second_optimizer.state[right]
            for key in ("exp_avg", "exp_avg_sq", "step"):
                self.assertTrue(torch.equal(left_state[key], right_state[key]))


if __name__ == "__main__":
    unittest.main()
