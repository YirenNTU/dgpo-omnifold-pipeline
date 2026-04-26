#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import awkward as ak

from export_evenet_prediction_to_qi import (
    build_predicted_reconstruction,
    prepare_events_for_parquet,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect one prediction-row through the export reconstruction pipeline, "
            "showing lead_a/lead_b p4 before and after parquet materialization."
        )
    )
    parser.add_argument("--prediction-parquet", type=Path, required=True)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument(
        "--pred-class-name",
        type=str,
        default=None,
        help="Optional evenet_pred_class_name filter, e.g. Ztautau_pipi.",
    )
    parser.add_argument(
        "--require-valid",
        action="store_true",
        help="If set, only inspect rows with flags_valid=True after reconstruction.",
    )
    return parser.parse_args()


def p4_components_from_record(record: ak.Record, field: str) -> dict[str, float]:
    p4 = record[field]
    fields = set(getattr(p4, "fields", []))
    if {"px", "py", "pz", "E"}.issubset(fields):
        px = float(p4["px"])
        py = float(p4["py"])
        pz = float(p4["pz"])
        energy = float(p4["E"])
    elif {"x", "y", "z", "t"}.issubset(fields):
        px = float(p4["x"])
        py = float(p4["y"])
        pz = float(p4["z"])
        energy = float(p4["t"])
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


def format_block(label: str, values: dict[str, float]) -> str:
    return "\n".join(
        [
            label,
            f"  px ={values['px']: .8f}",
            f"  py ={values['py']: .8f}",
            f"  pz ={values['pz']: .8f}",
            f"  E  ={values['E']: .8f}",
            f"  pt ={values['pt']: .8f}",
            f"  eta={values['eta']: .8f}",
            f"  phi={values['phi']: .8f}",
        ]
    )


def main() -> None:
    args = parse_args()
    pred_events = ak.from_parquet(args.prediction_parquet)
    pred_values, _ = build_predicted_reconstruction(pred_events, pred_events)
    pre_events = ak.Array(pred_values)
    post_events = prepare_events_for_parquet(pre_events)

    if args.pred_class_name is not None or args.require_valid:
        mask = np.ones(len(pred_events), dtype=bool)
        if args.pred_class_name is not None:
            class_names = np.asarray(ak.to_list(pred_events["evenet_pred_class_name"]), dtype=object)
            mask &= class_names == args.pred_class_name
        if args.require_valid:
            mask &= ak.to_numpy(pre_events["flags_valid"], allow_missing=False).astype(bool)
        matching_indices = np.flatnonzero(mask)
        if int(args.row) < 0 or int(args.row) >= len(matching_indices):
            raise IndexError(
                f"Requested row={args.row}, but only {len(matching_indices)} rows match "
                f"pred_class_name={args.pred_class_name!r}, require_valid={args.require_valid}."
            )
        index = int(matching_indices[int(args.row)])
    else:
        index = int(args.row)

    pred_event = pred_events[index]
    pre = pre_events[index]
    post = post_events[index]

    print(f"file: {args.prediction_parquet}")
    print(f"row={index}")
    if args.pred_class_name is not None or args.require_valid:
        print(
            f"selection: pred_class_name={args.pred_class_name!r}, "
            f"require_valid={args.require_valid}, selected_match_index={args.row}"
        )
    print(f"source_slot_for_a={int(pred_event['source_slot_for_a'])}")
    print(f"source_slot_for_b={int(pred_event['source_slot_for_b'])}")
    print(f"evenet_slot_for_a={int(pre['evenet_slot_for_a'])}")
    print(f"evenet_slot_for_b={int(pre['evenet_slot_for_b'])}")
    print(f"evenet_pred_class_name={pred_event['evenet_pred_class_name']}")
    print(f"evenet_truth_class_name={pred_event['evenet_truth_class_name']}")
    print(f"flags_valid={bool(pre['flags_valid'])}")
    print()

    for field in (
        "lead_a_visible_p4",
        "lead_b_visible_p4",
        "lead_a_missing_p4",
        "lead_b_missing_p4",
        "reco_tau_a_p4",
        "reco_tau_b_p4",
    ):
        print(format_block(f"{field} [pre-materialize]", p4_components_from_record(pre, field)))
        print(format_block(f"{field} [post-materialize]", p4_components_from_record(post, field)))
        print()


if __name__ == "__main__":
    main()
