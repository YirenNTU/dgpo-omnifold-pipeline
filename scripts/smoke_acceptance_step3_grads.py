#!/usr/bin/env python3
"""Fast CPU smoke: acceptance step≈3 finite-fwd / NaN-bwd without OmniFold folds.

Exercises the decoder graph that fails on NERSC acceptance audit:
AdaLN gates open, empty/single-visible memory, peaked mean-one weights,
head_dropout=0.25 (FFN only; MHA attn dropout must stay 0).

  PYTHONPATH=evenet_dgpo:evenet_dgpo/RL python3 scripts/smoke_acceptance_step3_grads.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

import importlib.util
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evenet_dgpo"), str(ROOT / "evenet_dgpo" / "RL")]

# Import evenet_ratio without omnifold_ztautau/__init__.py (avoids lightning).
_PKG = "RL.DGPO_neutrino.omnifold_ztautau"
_DIR = ROOT / "evenet_dgpo" / "RL" / "DGPO_neutrino" / "omnifold_ztautau"
for _name in ("RL", "RL.DGPO_neutrino", _PKG):
    if _name not in sys.modules:
        _mod = types.ModuleType(_name)
        _mod.__path__ = [str(_DIR)] if _name == _PKG else []
        sys.modules[_name] = _mod


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_load(f"{_PKG}.rest_frame", _DIR / "rest_frame.py")
_er = _load(f"{_PKG}.evenet_ratio", _DIR / "evenet_ratio.py")
AdaLNZeroCandidateDecoder = _er.AdaLNZeroCandidateDecoder
AdaLNZeroDecoderBlock = _er.AdaLNZeroDecoderBlock


def main() -> int:
    block = AdaLNZeroDecoderBlock(8, 2, 0.25, 8)
    if float(block.self_attn.dropout) != 0.0 or float(block.cross_attn.dropout) != 0.0:
        print(
            "FAIL: decoder MHA dropout must be 0 "
            f"(self={block.self_attn.dropout}, cross={block.cross_attn.dropout})"
        )
        return 1
    if float(block.ffn[2].p) != 0.25:
        print(f"FAIL: FFN dropout should remain 0.25, got {block.ffn[2].p}")
        return 1

    batch, hidden, memory_slots = 4, 8, 6
    decoder = AdaLNZeroCandidateDecoder(
        token_dim=hidden,
        event_dim=hidden,
        hidden_dim=hidden,
        num_layers=1,
        num_heads=2,
        dropout=0.25,
    )
    for layer in decoder.blocks:
        nn.init.normal_(layer.modulation.proj.weight, std=0.05)
        nn.init.normal_(layer.modulation.proj.bias, std=0.05)
    decoder.train()

    candidates = torch.randn(batch, 2, hidden, requires_grad=True)
    memory = torch.randn(batch, memory_slots, hidden)
    # Sparse cross-attn keys: only sentinel-like last key, plus one extra on row 2.
    memory_mask = torch.zeros(batch, memory_slots, 1, dtype=torch.bool)
    memory_mask[:, -1] = True
    memory_mask[2, 0] = True
    event = torch.randn(batch, hidden)
    raw = torch.tensor([1.0e-3, 1.0e3, 1.0, 1.0])
    weight = raw * (raw.numel() / raw.sum())

    for seed in range(8):
        torch.manual_seed(seed)
        decoder.zero_grad(set_to_none=True)
        if candidates.grad is not None:
            candidates.grad = None
        out = decoder(
            candidate_tokens=candidates,
            event_token=event,
            memory_tokens=memory,
            memory_mask=memory_mask,
        )
        loss = (weight[:, None, None] * out.square()).mean()
        loss.backward()
        if not torch.isfinite(loss):
            print(f"FAIL: nonfinite loss at seed={seed}")
            return 1
        bad = [
            name
            for name, parameter in decoder.named_parameters()
            if parameter.grad is not None
            and not bool(torch.isfinite(parameter.grad).all())
        ]
        if bad:
            print(f"FAIL: nonfinite grads at seed={seed}: {bad}")
            return 1
        if candidates.grad is None or not bool(torch.isfinite(candidates.grad).all()):
            print(f"FAIL: candidate input grads bad at seed={seed}")
            return 1

    # Cold Adam steps from zero gates: first updates open AdaLN like audit step 1–3.
    cold = AdaLNZeroCandidateDecoder(
        token_dim=hidden,
        event_dim=hidden,
        hidden_dim=hidden,
        num_layers=1,
        num_heads=2,
        dropout=0.25,
    )
    cold.train()
    opt = torch.optim.AdamW(cold.parameters(), lr=1.0e-3)
    for step in range(3):
        opt.zero_grad(set_to_none=True)
        out = cold(
            candidate_tokens=torch.randn(batch, 2, hidden),
            event_token=torch.randn(batch, hidden),
            memory_tokens=memory.detach(),
            memory_mask=memory_mask,
        )
        loss = (weight[:, None, None] * out.square()).mean()
        loss.backward()
        bad = [
            name
            for name, parameter in cold.named_parameters()
            if parameter.grad is not None
            and not bool(torch.isfinite(parameter.grad).all())
        ]
        if bad:
            print(f"FAIL: cold-start nonfinite grads at step={step + 1}: {bad}")
            return 1
        opt.step()

    print(
        "PASS: decoder attn_dropout=0; open-gate sparse+peaked and "
        "cold 3-step Adam grads finite"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
