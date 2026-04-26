#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import awkward as ak
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect one exported EveNet row and compare slot0/slot1 kinematics "
            "against stored lead_a/lead_b p4 fields."
        )
    )
    parser.add_argument("--export-parquet", type=Path, required=True)
    parser.add_argument(
        "--predicted-row",
        type=int,
        default=0,
        help="Row index inside the evenet_has_prediction==True subset. Default: 0",
    )
    parser.add_argument(
        "--absolute-row",
        type=int,
        default=None,
        help="Optional absolute row index in the parquet. Overrides --predicted-row.",
    )
    return parser.parse_args()


def to_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def export_p4_components(event: ak.Record, field: str) -> dict[str, float]:
    p4 = event[field]
    fields = set(getattr(p4, "fields", []))
    if {"x", "y", "z", "t"}.issubset(fields):
        px = float(p4["x"])
        py = float(p4["y"])
        pz = float(p4["z"])
        energy = float(p4["t"])
    elif {"px", "py", "pz", "E"}.issubset(fields):
        px = float(p4["px"])
        py = float(p4["py"])
        pz = float(p4["pz"])
        energy = float(p4["E"])
    else:
        raise KeyError(f"{field} has unsupported fields: {sorted(fields)}")

    pt = math.sqrt(px * px + py * py)
    phi = math.atan2(py, px)
    eta = math.asinh(pz / pt) if pt > 0 else float("nan")
    return {
        "px": px,
        "py": py,
        "pz": pz,
        "E": energy,
        "pt": pt,
        "eta": eta,
        "phi": phi,
    }


def expected_from_slot(event: ak.Record, prefix: str, slot: int) -> dict[str, float]:
    energy = float(event[f"{prefix}_slot{slot}_energy"])
    pt = float(event[f"{prefix}_slot{slot}_pt"])
    eta = float(event[f"{prefix}_slot{slot}_eta"])
    phi = float(event[f"{prefix}_slot{slot}_phi"])
    return {
        "px": pt * math.cos(phi),
        "py": pt * math.sin(phi),
        "pz": pt * math.sinh(eta),
        "E": energy,
        "pt": pt,
        "eta": eta,
        "phi": phi,
    }


def format_component_block(name: str, expected: dict[str, float], stored: dict[str, float]) -> str:
    lines = [f"{name}"]
    for key in ("px", "py", "pz", "E", "pt", "eta", "phi"):
        exp_value = expected[key]
        stored_value = stored[key]
        lines.append(
            f"  {key:<3} expected={exp_value: .8f}  stored={stored_value: .8f}  diff={stored_value - exp_value: .8e}"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    events = ak.from_parquet(args.export_parquet)

    if args.absolute_row is not None:
        event = events[int(args.absolute_row)]
        row_label = f"absolute_row={int(args.absolute_row)}"
    else:
        if "evenet_has_prediction" not in events.fields:
            raise KeyError(f"{args.export_parquet} is missing evenet_has_prediction")
        predicted = events[events["evenet_has_prediction"] == 1]
        event = predicted[int(args.predicted_row)]
        row_label = f"predicted_row={int(args.predicted_row)}"

    slot_for_a = int(event["source_slot_for_a"])
    slot_for_b = int(event["source_slot_for_b"])

    print(f"file: {args.export_parquet}")
    print(row_label)
    print(f"source_slot_for_a={slot_for_a}")
    print(f"source_slot_for_b={slot_for_b}")
    print(f"evenet_pred_class_name={to_scalar(event['evenet_pred_class_name'])}")
    print(f"evenet_truth_class_name={to_scalar(event['evenet_truth_class_name'])}")
    print(f"flags_valid={bool(event['flags_valid'])}")
    print("")

    for slot in (0, 1):
        print(f"slot{slot} visible:  E={float(event[f'tau_vis_prong_slot{slot}_energy']):.8f}  pt={float(event[f'tau_vis_prong_slot{slot}_pt']):.8f}  eta={float(event[f'tau_vis_prong_slot{slot}_eta']):.8f}  phi={float(event[f'tau_vis_prong_slot{slot}_phi']):.8f}")
        print(f"slot{slot} missing:  E={float(event[f'pred_invisible_slot{slot}_energy']):.8f}  pt={float(event[f'pred_invisible_slot{slot}_pt']):.8f}  eta={float(event[f'pred_invisible_slot{slot}_eta']):.8f}  phi={float(event[f'pred_invisible_slot{slot}_phi']):.8f}")
    print("")

    expected_a_visible = expected_from_slot(event, "tau_vis_prong", slot_for_a)
    expected_b_visible = expected_from_slot(event, "tau_vis_prong", slot_for_b)
    expected_a_missing = expected_from_slot(event, "pred_invisible", slot_for_a)
    expected_b_missing = expected_from_slot(event, "pred_invisible", slot_for_b)

    stored_a_visible = export_p4_components(event, "lead_a_visible_p4")
    stored_b_visible = export_p4_components(event, "lead_b_visible_p4")
    stored_a_missing = export_p4_components(event, "lead_a_missing_p4")
    stored_b_missing = export_p4_components(event, "lead_b_missing_p4")

    print(format_component_block("lead_a_visible_p4", expected_a_visible, stored_a_visible))
    print("")
    print(format_component_block("lead_b_visible_p4", expected_b_visible, stored_b_visible))
    print("")
    print(format_component_block("lead_a_missing_p4", expected_a_missing, stored_a_missing))
    print("")
    print(format_component_block("lead_b_missing_p4", expected_b_missing, stored_b_missing))


if __name__ == "__main__":
    main()
