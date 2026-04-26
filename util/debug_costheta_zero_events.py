#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import vector


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quantum.observables_builder import build_observables, get_observable_names, helicity_basis
from utils.common_functions import rebuild_p4


vector.register_awkward()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print detailed event-level diagnostics for events where a reco cos(theta) observable "
            "is close to zero."
        )
    )
    parser.add_argument("--parquet", type=Path, required=True, help="Export parquet to inspect.")
    parser.add_argument(
        "--observable",
        required=True,
        choices=get_observable_names(),
        help="Reco observable to inspect, e.g. cos_theta_B_k.",
    )
    parser.add_argument(
        "--source",
        choices=["stored", "recompute"],
        default="recompute",
        help="Use stored scalar observable or recompute it from p4.",
    )
    parser.add_argument(
        "--abs-threshold",
        type=float,
        default=0.02,
        help="Select events with |observable| <= threshold. Default: 0.02",
    )
    parser.add_argument("--max-events", type=int, default=10, help="Maximum events to print.")
    parser.add_argument(
        "--require-valid",
        action="store_true",
        help="Require flags_valid=True when selecting candidate events.",
    )
    parser.add_argument(
        "--pred-class-name",
        type=str,
        default=None,
        help="Optional evenet_pred_class_name filter.",
    )
    parser.add_argument(
        "--truth-class-name",
        type=str,
        default=None,
        help="Optional evenet_truth_class_name filter.",
    )
    return parser.parse_args()


def load_events(path: Path) -> ak.Array:
    events = ak.from_parquet(path)
    for field in events.fields:
        if field.endswith("_p4"):
            events[field] = rebuild_vector(events[field])
    return events


def rebuild_vector(values: ak.Array) -> ak.Array:
    fields = set(getattr(values, "fields", []))
    if {"px", "py", "pz", "E"}.issubset(fields):
        return vector.zip({"px": values["px"], "py": values["py"], "pz": values["pz"], "E": values["E"]})
    if {"x", "y", "z", "t"}.issubset(fields):
        return rebuild_p4(values)
    return values


def to_numpy(values: Any, dtype=np.float64) -> np.ndarray:
    return ak.to_numpy(values, allow_missing=False).astype(dtype)


def to_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def p4_dict(p4: Any) -> dict[str, float]:
    return {
        "px": float(p4.px),
        "py": float(p4.py),
        "pz": float(p4.pz),
        "E": float(p4.E),
        "pt": float(p4.pt),
        "eta": float(p4.eta),
        "phi": float(p4.phi),
        "mass": float(p4.mass),
    }


def one_event_vector_array(event: ak.Record, field: str) -> ak.Array:
    p4 = event[field]
    fields = set(getattr(p4, "fields", []))
    if {"px", "py", "pz", "E"}.issubset(fields):
        return vector.zip(
            {
                "px": ak.Array([float(p4["px"])]),
                "py": ak.Array([float(p4["py"])]),
                "pz": ak.Array([float(p4["pz"])]),
                "E": ak.Array([float(p4["E"])]),
            }
        )
    if {"x", "y", "z", "t"}.issubset(fields):
        return vector.zip(
            {
                "px": ak.Array([float(p4["x"])]),
                "py": ak.Array([float(p4["y"])]),
                "pz": ak.Array([float(p4["z"])]),
                "E": ak.Array([float(p4["t"])]),
            }
        )
    raise KeyError(f"{field} has unsupported p4 fields: {sorted(fields)}")


def vec3_dict(v: Any) -> dict[str, float]:
    return {
        "x": float(v.x),
        "y": float(v.y),
        "z": float(v.z),
        "mag": float(np.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)),
    }


def format_mapping(label: str, mapping: dict[str, float]) -> str:
    order = ["px", "py", "pz", "E", "pt", "eta", "phi", "mass"] if "px" in mapping else ["x", "y", "z", "mag"]
    parts = [f"{key}={mapping[key]: .8f}" for key in order if key in mapping]
    return f"{label}: " + "  ".join(parts)


def observable_values(events: ak.Array, observable: str, source: str) -> np.ndarray | None:
    if source == "stored":
        if observable not in events.fields:
            return None
        return to_numpy(events[observable], np.float64)

    required = {"reco_tau_a_p4", "reco_tau_b_p4", "lead_a_visible_p4", "lead_b_visible_p4"}
    if not required.issubset(set(events.fields)):
        return None
    observables = build_observables(
        events["reco_tau_a_p4"],
        events["reco_tau_b_p4"],
        events["lead_a_visible_p4"],
        events["lead_b_visible_p4"],
    )
    if observable not in observables:
        return None
    return to_numpy(observables[observable], np.float64)


def nominal_geometry(event: ak.Record) -> dict[str, Any]:
    tau_a = event["reco_tau_a_p4"]
    tau_b = event["reco_tau_b_p4"]
    vis_a = event["lead_a_visible_p4"]
    vis_b = event["lead_b_visible_p4"]

    cm = tau_a + tau_b
    boost_to_cm = -cm.to_beta3()

    tau_a_cm = tau_a.boost(boost_to_cm)
    tau_b_cm = tau_b.boost(boost_to_cm)
    vis_a_cm = vis_a.boost(boost_to_cm)
    vis_b_cm = vis_b.boost(boost_to_cm)

    basis_cm = helicity_basis(tau_a_cm)

    boost_to_a_rest = -tau_a_cm.to_beta3()
    boost_to_b_rest = -tau_b_cm.to_beta3()
    vis_a_rest = vis_a_cm.boost(boost_to_a_rest).to_pxpypz().unit()
    vis_b_rest = vis_b_cm.boost(boost_to_b_rest).to_pxpypz().unit()

    basis_a_rest = {}
    basis_b_rest = {}
    dots_a = {}
    dots_b = {}
    for axis in ["n", "r", "k"]:
        axis_vector = basis_cm[axis]
        basis_tmp = vector.zip(
            {
                "x": axis_vector.x,
                "y": axis_vector.y,
                "z": axis_vector.z,
                "t": np.zeros_like(axis_vector.x),
            }
        )
        basis_a_rest[axis] = basis_tmp.boost(boost_to_a_rest).to_pxpypz().unit()
        basis_b_rest[axis] = basis_tmp.boost(boost_to_b_rest).to_pxpypz().unit()
        dots_a[axis] = float(vis_a_rest.dot(basis_a_rest[axis]))
        dots_b[axis] = float(vis_b_rest.dot(basis_b_rest[axis]))

    return {
        "tau_a_cm": tau_a_cm,
        "tau_b_cm": tau_b_cm,
        "vis_a_cm": vis_a_cm,
        "vis_b_cm": vis_b_cm,
        "vis_a_rest": vis_a_rest,
        "vis_b_rest": vis_b_rest,
        "basis_cm": basis_cm,
        "basis_a_rest": basis_a_rest,
        "basis_b_rest": basis_b_rest,
        "dots_a": dots_a,
        "dots_b": dots_b,
    }


def print_event(event: ak.Record, index: int, observable: str, source: str) -> None:
    print("=" * 100)
    print(f"event_index={index} observable={observable} source={source}")
    for field in (
        "flags_valid",
        "weight",
        "central_weight",
        "evenet_weight",
        "evenet_pred_class_name",
        "evenet_truth_class_name",
        "source_slot_for_a",
        "source_slot_for_b",
        "evenet_slot_for_a",
        "evenet_slot_for_b",
    ):
        if field in event.fields:
            print(f"{field}={to_scalar(event[field])}")
    if observable in event.fields:
        print(f"stored_{observable}={float(event[observable]): .8f}")
    truth_field = f"truth_{observable}"
    if truth_field in event.fields:
        print(f"{truth_field}={float(event[truth_field]): .8f}")

    for field in ("lead_a_visible_p4", "lead_b_visible_p4", "lead_a_missing_p4", "lead_b_missing_p4", "reco_tau_a_p4", "reco_tau_b_p4"):
        if field in event.fields:
            print(format_mapping(field, p4_dict(event[field])))

    geom = nominal_geometry(event)
    print(format_mapping("tau_a_cm", p4_dict(geom["tau_a_cm"])))
    print(format_mapping("tau_b_cm", p4_dict(geom["tau_b_cm"])))
    print(format_mapping("vis_a_rest_unit", vec3_dict(geom["vis_a_rest"])))
    print(format_mapping("vis_b_rest_unit", vec3_dict(geom["vis_b_rest"])))
    for axis in ["n", "r", "k"]:
        print(format_mapping(f"basis_cm_{axis}", vec3_dict(geom["basis_cm"][axis])))
        print(format_mapping(f"basis_a_rest_{axis}", vec3_dict(geom["basis_a_rest"][axis])))
        print(format_mapping(f"basis_b_rest_{axis}", vec3_dict(geom["basis_b_rest"][axis])))
        print(f"dot_A_{axis}={geom['dots_a'][axis]: .8f}")
        print(f"dot_B_{axis}={geom['dots_b'][axis]: .8f}")

    observables = build_observables(
        one_event_vector_array(event, "reco_tau_a_p4"),
        one_event_vector_array(event, "reco_tau_b_p4"),
        one_event_vector_array(event, "lead_a_visible_p4"),
        one_event_vector_array(event, "lead_b_visible_p4"),
    )
    print(f"recomputed_{observable}={float(ak.to_numpy(observables[observable], allow_missing=False)[0]): .8f}")


def main() -> None:
    args = parse_args()
    events = load_events(args.parquet)
    values = observable_values(events, args.observable, args.source)
    if values is None:
        raise ValueError(
            f"Could not obtain observable '{args.observable}' from {args.parquet} with source={args.source}."
        )

    mask = np.isfinite(values) & (np.abs(values) <= float(args.abs_threshold))
    if args.require_valid and "flags_valid" in events.fields:
        mask &= to_numpy(events["flags_valid"], np.bool_)
    if args.pred_class_name is not None and "evenet_pred_class_name" in events.fields:
        pred_classes = np.asarray(ak.to_list(events["evenet_pred_class_name"]), dtype=object)
        mask &= pred_classes == args.pred_class_name
    if args.truth_class_name is not None and "evenet_truth_class_name" in events.fields:
        truth_classes = np.asarray(ak.to_list(events["evenet_truth_class_name"]), dtype=object)
        mask &= truth_classes == args.truth_class_name

    indices = np.flatnonzero(mask)
    print(f"file={args.parquet}")
    print(f"observable={args.observable}")
    print(f"source={args.source}")
    print(f"abs_threshold={args.abs_threshold}")
    print(f"require_valid={args.require_valid}")
    print(f"pred_class_name={args.pred_class_name!r}")
    print(f"truth_class_name={args.truth_class_name!r}")
    print(f"matching_events={len(indices)}")

    if len(indices) == 0:
        return

    for index in indices[: int(args.max_events)]:
        print_event(events[int(index)], int(index), args.observable, args.source)


if __name__ == "__main__":
    main()
