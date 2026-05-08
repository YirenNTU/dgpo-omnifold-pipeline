#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path
from typing import Any

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import vector


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quantum.observables_builder import build_observables, get_observable_names, helicity_basis
from utils.common_functions import rebuild_p4


vector.register_awkward()

TAU_MASS = 1.777  # GeV
CM_ENERGY = 91.2 # GeV



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare stored truth observables against observables rebuilt from "
            "target or predicted missing plus visible tau inputs."
        )
    )
    parser.add_argument(
        "--parquet",
        nargs="+",
        required=True,
        help="Input parquet file(s), directory/directories, or glob pattern(s).",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for plots and JSON summary.")
    parser.add_argument(
        "--observables",
        nargs="+",
        default=None,
        help="Optional observable subset. Defaults to theta_cm, mtautau, and all cos_theta_*.",
    )
    parser.add_argument("--batch-size", type=int, default=50000, help="Rows per parquet streaming batch.")
    parser.add_argument("--max-entries", type=int, default=None, help="Optional global row cap.")
    parser.add_argument("--region", default=None, help="Optional region name; if <region>_cut exists, keep only that selection.")
    parser.add_argument(
        "--truth-region-only",
        action="store_true",
        help="Require truth_QI_region == 1 when that field exists.",
    )
    parser.add_argument(
        "--weight-field",
        choices=["auto", "weight", "central_weight", "evenet_weight", "unit"],
        default="unit",
        help="Weight source for metrics and 1D histograms.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize the 1D histograms to unit area.",
    )
    parser.add_argument(
        "--debug-chain",
        action="store_true",
        help=(
            "Write extra Target-vs-Predicted debug-chain plots for remapped neutrino inputs, "
            "mass-projected taus, and derived observables."
        ),
    )
    return parser.parse_args()


def sanitize_filename(name: str) -> str:
    chars = [char if char.isalnum() or char in {"_", "-", "."} else "_" for char in name]
    return "".join(chars).strip("_")


def resolve_parquet_inputs(items: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in items:
        expanded = Path(item).expanduser()
        if expanded.is_dir():
            paths.extend(sorted(expanded.glob("*.parquet")))
            continue
        matches = sorted(glob.glob(str(expanded)))
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(expanded)
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    if not unique:
        raise FileNotFoundError("No parquet inputs found.")
    return unique


def requested_observables(raw: list[str] | None) -> list[str]:
    return list(raw) if raw else list(get_observable_names())


def rebuild_vector(values: ak.Array) -> ak.Array:
    fields = set(values.fields)
    if {"px", "py", "pz", "E"}.issubset(fields):
        return vector.zip({"px": values["px"], "py": values["py"], "pz": values["pz"], "E": values["E"]})
    if {"x", "y", "z", "t"}.issubset(fields):
        return rebuild_p4(values)
    return values


def to_numpy(values: Any, dtype=np.float64) -> np.ndarray:
    return ak.to_numpy(values, allow_missing=False).astype(dtype)


def build_momentum4d(px: np.ndarray, py: np.ndarray, pz: np.ndarray, energy: np.ndarray) -> ak.Array:
    return ak.zip(
        {
            "px": np.asarray(px, dtype=np.float64),
            "py": np.asarray(py, dtype=np.float64),
            "pz": np.asarray(pz, dtype=np.float64),
            "E": np.asarray(energy, dtype=np.float64),
        },
        with_name="Momentum4D",
    )


def build_momentum4d_with_mass(obj: ak.Array, mass: float, E: float=None) -> ak.Array:
    px = np.asarray(obj.px, dtype=np.float64)
    py = np.asarray(obj.py, dtype=np.float64)
    pz = np.asarray(obj.pz, dtype=np.float64)
    energy = np.sqrt(px * px + py * py + pz * pz + mass * mass)
    return build_momentum4d(px, py, pz, energy)

def build_momentum4d_with_energy_mass(obj: ak.Array, energy: float, mass: float) -> ak.Array:
    eta = np.asarray(obj.eta, dtype=np.float64)
    phi = np.asarray(obj.phi, dtype=np.float64)

    if energy <= mass:
        raise ValueError(f"energy must be larger than mass, got energy={energy}, mass={mass}")

    p = np.sqrt(energy * energy - mass * mass)
    pt = p / np.cosh(eta)

    return ak.zip(
        {
            "pt": pt,
            "eta": eta,
            "phi": phi,
            "mass": np.full_like(eta, mass, dtype=np.float64),
        },
        with_name="Momentum4D",
    )

def build_tau_tau_pair(tau_a: ak.Array, tau_b: ak.Array) -> ak.Array:
    energy = CM_ENERGY / 2
    mass = TAU_MASS
    p = (energy*energy - mass*mass)**0.5

    # reconstruct pt

    pt_a = p / np.cosh(tau_a.eta)
    pt_b = p / np.cosh(tau_b.eta)

    px_a = pt_a * np.cos(tau_a.phi)
    py_a = pt_a * np.sin(tau_a.phi)
    pz_a = pt_a * np.cosh(tau_a.eta)
    px_b = pt_b * np.cos(tau_b.phi)
    py_b = pt_b * np.sin(tau_b.phi)
    pz_b = pt_b * np.cosh(tau_b.eta)

    px_shift = (px_a + px_b)
    py_shift = (py_a + py_b)
    pz_shift = (pz_b + pz_b)

    px_a_corrected = px_a - (px_shift/2)
    px_b_corrected = px_b - (px_shift/2)

    py_a_corrected = py_a - (py_shift / 2)
    py_b_corrected = py_b - (py_shift / 2)

    pz_a_corrected = pz_a - (pz_shift/2)
    pz_b_corrected = pz_b - (pz_shift/2)


    tau_a_reco = ak.zip(
        {
            "px": px_a_corrected,
            "py": py_a_corrected,
            "pz": px_a_corrected,
            "E": np.full_like(px_a_corrected, energy, dtype=np.float64),
        },
        with_name="Momentum4D",
    )
    tau_b_reco = ak.zip(
        {
            "px": px_b_corrected,
            "py": py_b_corrected,
            "pz": px_b_corrected,
            "E": np.full_like(px_b_corrected, energy, dtype=np.float64),
        },
        with_name="Momentum4D",
    )

    return tau_a_reco, tau_b_reco

def massless_p4_from_pt_eta_phi(pt: np.ndarray, eta: np.ndarray, phi: np.ndarray) -> ak.Array:
    return build_momentum4d(
        pt * np.cos(phi),
        pt * np.sin(phi),
        pt * np.sinh(eta),
        pt * np.cosh(eta),
    )


def choose_component_by_slot(events: ak.Array, prefix: str, slot_indices: np.ndarray, component: str) -> np.ndarray | None:
    slot0_name = f"{prefix}_slot0_{component}"
    slot1_name = f"{prefix}_slot1_{component}"
    fields = set(events.fields)
    if slot0_name not in fields or slot1_name not in fields:
        return None
    slot0 = to_numpy(events[slot0_name], np.float64)
    slot1 = to_numpy(events[slot1_name], np.float64)
    return np.where(slot_indices == 0, slot0, slot1)


def p4_from_slot_features(events: ak.Array, prefix: str, slot: int) -> ak.Array:
    energy = to_numpy(events[f"{prefix}_slot{slot}_energy"], np.float64)
    pt = to_numpy(events[f"{prefix}_slot{slot}_pt"], np.float64)
    eta = to_numpy(events[f"{prefix}_slot{slot}_eta"], np.float64)
    phi = to_numpy(events[f"{prefix}_slot{slot}_phi"], np.float64)
    return build_momentum4d(
        pt * np.cos(phi),
        pt * np.sin(phi),
        pt * np.sinh(eta),
        energy,
    )


def massless_p4_from_slot_features(events: ak.Array, prefix: str, slot: int) -> ak.Array:
    pt = to_numpy(events[f"{prefix}_slot{slot}_pt"], np.float64)
    eta = to_numpy(events[f"{prefix}_slot{slot}_eta"], np.float64)
    phi = to_numpy(events[f"{prefix}_slot{slot}_phi"], np.float64)
    return massless_p4_from_pt_eta_phi(pt, eta, phi)


def remapped_slot_p4(events: ak.Array, prefix: str, leg: str) -> ak.Array | None:
    slot_field = f"source_slot_for_{leg}"
    if slot_field not in events.fields:
        return None
    slot_indices = to_numpy(events[slot_field], np.int64)

    px = choose_component_by_slot(events, prefix, slot_indices, "px")
    py = choose_component_by_slot(events, prefix, slot_indices, "py")
    pz = choose_component_by_slot(events, prefix, slot_indices, "pz")
    energy = choose_component_by_slot(events, prefix, slot_indices, "E")
    if px is not None and py is not None and pz is not None and energy is not None:
        return build_momentum4d(px, py, pz, energy)

    energy = choose_component_by_slot(events, prefix, slot_indices, "energy")
    pt = choose_component_by_slot(events, prefix, slot_indices, "pt")
    eta = choose_component_by_slot(events, prefix, slot_indices, "eta")
    phi = choose_component_by_slot(events, prefix, slot_indices, "phi")
    if energy is not None and pt is not None and eta is not None and phi is not None:
        return build_momentum4d(
            pt * np.cos(phi),
            pt * np.sin(phi),
            pt * np.sinh(eta),
            energy,
        )
    if pt is not None and eta is not None and phi is not None:
        return massless_p4_from_pt_eta_phi(pt, eta, phi)
    return None


def remapped_slot_component(events: ak.Array, prefix: str, leg: str, component: str) -> np.ndarray:
    slot_field = f"source_slot_for_{leg}"
    if slot_field not in events.fields:
        raise ValueError(f"Missing slot mapping field '{slot_field}'.")
    slot_indices = to_numpy(events[slot_field], np.int64)
    values = choose_component_by_slot(events, prefix, slot_indices, component)
    if values is None:
        raise ValueError(f"Missing component '{component}' for prefix '{prefix}'.")
    return values


def visible_tau_p4(events: ak.Array, leg: str) -> ak.Array:
    fields = set(events.fields)
    direct_field = f"lead_{leg}_visible_p4"
    if direct_field in fields:
        return rebuild_vector(events[direct_field])
    remapped = remapped_slot_p4(events, "tau_vis_prong", leg)
    if remapped is not None:
        return remapped
    raise ValueError(f"Missing visible-tau fields for leg '{leg}'.")


def target_missing_p4(events: ak.Array, leg: str) -> ak.Array:
    remapped = remapped_slot_p4(events, "target_invisible", leg)
    if remapped is not None:
        return remapped
    raise ValueError(f"Missing target-invisible fields for leg '{leg}'.")


def predicted_missing_p4(events: ak.Array, leg: str) -> ak.Array:
    remapped = remapped_slot_p4(events, "pred_invisible", leg)
    if remapped is not None:
        return remapped
    raise ValueError(f"Missing pred-invisible fields for leg '{leg}'.")


def flattened_leg_feature(events: ak.Array, prefix: str, component: str) -> np.ndarray:
    values = []
    for leg in ("a", "b"):
        values.append(remapped_slot_component(events, prefix, leg, component))
    return np.concatenate(values)


def parquet_columns_to_load(path: Path, observables: list[str], region: str | None, truth_region_only: bool) -> list[str] | None:
    schema = pq.read_schema(path)
    available = {field.name for field in schema}
    columns: set[str] = set()
    for observable in observables:
        columns.add(f"truth_{observable}")
    for slot in (0, 1):
        columns.update(
            {
                f"target_invisible_slot{slot}_pt",
                f"target_invisible_slot{slot}_eta",
                f"target_invisible_slot{slot}_phi",
                f"pred_invisible_slot{slot}_pt",
                f"pred_invisible_slot{slot}_eta",
                f"pred_invisible_slot{slot}_phi",
                f"tau_vis_prong_slot{slot}_energy",
                f"tau_vis_prong_slot{slot}_pt",
                f"tau_vis_prong_slot{slot}_eta",
                f"tau_vis_prong_slot{slot}_phi",
            }
        )
    columns.update({"source_slot_for_a", "source_slot_for_b"})
    columns.update({"lead_a_visible_p4", "lead_b_visible_p4", "weight", "central_weight", "evenet_weight"})
    if region is not None:
        columns.add(f"{region}_cut")
    if truth_region_only:
        columns.add("truth_QI_region")
    selected = [name for name in sorted(columns) if name in available]
    return selected if selected else None


def iter_event_batches(
    path: Path,
    observables: list[str],
    batch_size: int,
    max_entries: int | None,
    region: str | None,
    truth_region_only: bool,
):
    columns = parquet_columns_to_load(path, observables, region, truth_region_only)
    parquet = pq.ParquetFile(path)
    remaining = None if max_entries is None else max(0, int(max_entries))
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        events = ak.from_arrow(batch)
        for field in events.fields:
            if field.endswith("_p4"):
                events[field] = rebuild_vector(events[field])
        if remaining is not None and len(events) > remaining:
            events = events[:remaining]
        if len(events) == 0:
            continue
        yield events
        if remaining is not None:
            remaining -= len(events)
            if remaining <= 0:
                break


def apply_event_selection(events: ak.Array, region: str | None, truth_region_only: bool) -> ak.Array:
    mask = np.ones(len(events), dtype=bool)
    if region is not None:
        region_field = f"{region}_cut"
        if region_field in events.fields:
            mask &= to_numpy(events[region_field], np.int64) > 0
    if truth_region_only and "truth_QI_region" in events.fields:
        mask &= to_numpy(events["truth_QI_region"], np.int64) > 0
    return events[mask]


def event_weights(events: ak.Array, mode: str) -> np.ndarray:
    if mode == "unit":
        return np.ones(len(events), dtype=np.float64)
    if mode == "auto":
        candidates = ["weight", "central_weight", "evenet_weight"]
    else:
        candidates = [mode]
    for field in candidates:
        if field in events.fields:
            weights = to_numpy(events[field], np.float64)
            return np.where(np.isfinite(weights), weights, 0.0)
    return np.ones(len(events), dtype=np.float64)


def truth_values(events: ak.Array, observable: str) -> np.ndarray:
    return to_numpy(events[f"truth_{observable}"], np.float64)


def target_reconstructed_values(events: ak.Array, observable: str) -> np.ndarray:
    visible_a = visible_tau_p4(events, "a")
    visible_b = visible_tau_p4(events, "b")
    target_a = target_missing_p4(events, "a")
    target_b = target_missing_p4(events, "b")
    tau_a = build_momentum4d_with_mass(visible_a + target_a, TAU_MASS)
    tau_b = build_momentum4d_with_mass(visible_b + target_b, TAU_MASS)
    observables = build_observables(
        tau_a_p4=tau_a,
        tau_b_p4=tau_b,
        vis_a_p4=visible_a,
        vis_b_p4=visible_b,
    )
    return np.asarray(observables[observable], dtype=np.float64)


def predicted_reconstructed_values(events: ak.Array, observable: str) -> np.ndarray:
    visible_a = visible_tau_p4(events, "a")
    visible_b = visible_tau_p4(events, "b")
    pred_a = predicted_missing_p4(events, "a")
    pred_b = predicted_missing_p4(events, "b")
    # tau_a = build_momentum4d_with_mass(visible_a + pred_a, TAU_MASS)
    # tau_b = build_momentum4d_with_mass(visible_b + pred_b, TAU_MASS)

    # tau_a = build_momentum4d_with_energy_mass(visible_a + pred_a, CM_ENERGY/2, TAU_MASS)
    # tau_b = build_momentum4d_with_energy_mass(visible_b + pred_b, CM_ENERGY/2, TAU_MASS)
    #
    tau_a, tau_b = build_tau_tau_pair(visible_a + pred_a, visible_b + pred_b)
    #
    # tau_b = build_momentum4d(
    #     px=-tau_a.px,
    #     py=-tau_a.py,
    #     pz=-tau_a.pz,
    #     energy=tau_a.energy,
    # )

    observables = build_observables(
        tau_a_p4=tau_a,
        tau_b_p4=tau_b,
        vis_a_p4=visible_a,
        vis_b_p4=visible_b,
    )
    return np.asarray(observables[observable], dtype=np.float64)


def reconstructed_chain_values(events: ak.Array, missing_kind: str) -> dict[str, np.ndarray]:
    if missing_kind == "target":
        missing_a = target_missing_p4(events, "a")
        missing_b = target_missing_p4(events, "b")
        prefix = "target"
    elif missing_kind == "pred":
        missing_a = predicted_missing_p4(events, "a")
        missing_b = predicted_missing_p4(events, "b")
        prefix = "pred"
    else:
        raise ValueError(f"Unsupported missing kind '{missing_kind}'.")

    visible_a = visible_tau_p4(events, "a")
    visible_b = visible_tau_p4(events, "b")
    # tau_a = build_momentum4d_with_mass(visible_a + missing_a, TAU_MASS)
    # tau_a = build_momentum4d_with_energy_mass(visible_a + missing_a, CM_ENERGY/2, TAU_MASS)
    # print(f"replace b to be -a, {tau_a.x}, {tau_a.y}")
    # tau_b = build_momentum4d(px=-tau_a.x, py=-tau_a.y, pz=-tau_a.z, energy=tau_a.energy)
    # print("replace b to be -a [success]")
    tau_a, tau_b = build_tau_tau_pair(visible_a + missing_a, visible_b + missing_b)


    cm_p4 = tau_a + tau_b
    boost_to_cm = cm_p4.to_beta3()
    tau_a_cm = tau_a.boost(boost_to_cm)
    tau_b_cm = tau_b.boost(boost_to_cm)
    vis_a_cm = visible_a.boost(boost_to_cm)
    vis_b_cm = visible_b.boost(boost_to_cm)

    basis_cm = helicity_basis(tau_a_cm)

    boost_to_a_rest = -tau_a_cm.to_beta3()
    boost_to_b_rest = -tau_b_cm.to_beta3()
    vis_a_rest = vis_a_cm.boost(boost_to_a_rest).to_pxpypz().unit()
    vis_b_rest = vis_b_cm.boost(boost_to_b_rest).to_pxpypz().unit()

    basis_a_rest: dict[str, Any] = {}
    basis_b_rest: dict[str, Any] = {}
    for axis in ("n", "r", "k"):
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

    observables = build_observables(
        tau_a_p4=tau_a,
        tau_b_p4=tau_b,
        vis_a_p4=visible_a,
        vis_b_p4=visible_b,
    )
    values: dict[str, np.ndarray] = {
        f"{prefix}_nu_a_pt": np.asarray(missing_a.pt, dtype=np.float64),
        f"{prefix}_nu_a_eta": np.asarray(missing_a.eta, dtype=np.float64),
        f"{prefix}_nu_a_phi": np.asarray(missing_a.phi, dtype=np.float64),
        f"{prefix}_nu_b_pt": np.asarray(missing_b.pt, dtype=np.float64),
        f"{prefix}_nu_b_eta": np.asarray(missing_b.eta, dtype=np.float64),
        f"{prefix}_nu_b_phi": np.asarray(missing_b.phi, dtype=np.float64),
        f"{prefix}_tau_a_pt": np.asarray(tau_a.pt, dtype=np.float64),
        f"{prefix}_tau_a_eta": np.asarray(tau_a.eta, dtype=np.float64),
        f"{prefix}_tau_a_phi": np.asarray(tau_a.phi, dtype=np.float64),
        f"{prefix}_tau_a_mass": np.asarray(tau_a.mass, dtype=np.float64),
        f"{prefix}_tau_a_energy": np.asarray(tau_a.energy, dtype=np.float64),
        f"{prefix}_tau_b_pt": np.asarray(tau_b.pt, dtype=np.float64),
        f"{prefix}_tau_b_eta": np.asarray(tau_b.eta, dtype=np.float64),
        f"{prefix}_tau_b_phi": np.asarray(tau_b.phi, dtype=np.float64),
        f"{prefix}_tau_b_mass": np.asarray(tau_b.mass, dtype=np.float64),
        f"{prefix}_tau_b_energy": np.asarray(tau_b.energy, dtype=np.float64),
        f"{prefix}_tau_a_cm_costheta": np.asarray(tau_a_cm.costheta, dtype=np.float64),
        f"{prefix}_tau_b_cm_costheta": np.asarray(tau_b_cm.costheta, dtype=np.float64),
        f"{prefix}_tau_a_cm_phi": np.asarray(tau_a_cm.phi, dtype=np.float64),
        f"{prefix}_tau_b_cm_phi": np.asarray(tau_b_cm.phi, dtype=np.float64),
        f"{prefix}_vis_a_rest_x": np.asarray(vis_a_rest.x, dtype=np.float64),
        f"{prefix}_vis_a_rest_y": np.asarray(vis_a_rest.y, dtype=np.float64),
        f"{prefix}_vis_a_rest_z": np.asarray(vis_a_rest.z, dtype=np.float64),
        f"{prefix}_vis_b_rest_x": np.asarray(vis_b_rest.x, dtype=np.float64),
        f"{prefix}_vis_b_rest_y": np.asarray(vis_b_rest.y, dtype=np.float64),
        f"{prefix}_vis_b_rest_z": np.asarray(vis_b_rest.z, dtype=np.float64),
        f"{prefix}_tautau_px": np.asarray(cm_p4.x, dtype=np.float64),
        f"{prefix}_tautau_py": np.asarray(cm_p4.y, dtype=np.float64),
        f"{prefix}_tautau_pz": np.asarray(cm_p4.z, dtype=np.float64),
    }
    for axis in ("n", "r", "k"):
        values[f"{prefix}_basis_a_rest_{axis}_x"] = np.asarray(basis_a_rest[axis].x, dtype=np.float64)
        values[f"{prefix}_basis_a_rest_{axis}_y"] = np.asarray(basis_a_rest[axis].y, dtype=np.float64)
        values[f"{prefix}_basis_a_rest_{axis}_z"] = np.asarray(basis_a_rest[axis].z, dtype=np.float64)
        values[f"{prefix}_basis_b_rest_{axis}_x"] = np.asarray(basis_b_rest[axis].x, dtype=np.float64)
        values[f"{prefix}_basis_b_rest_{axis}_y"] = np.asarray(basis_b_rest[axis].y, dtype=np.float64)
        values[f"{prefix}_basis_b_rest_{axis}_z"] = np.asarray(basis_b_rest[axis].z, dtype=np.float64)
    for observable_name, observable_values in observables.items():
        values[f"{prefix}_{observable_name}"] = np.asarray(observable_values, dtype=np.float64)
    return values


def observable_limits(observable: str, *value_arrays: np.ndarray) -> tuple[float, float]:
    if observable == "theta_cm":
        return 0.0, 1.0
    if observable.startswith("cos_theta_"):
        return -1.0, 1.0
    merged = np.concatenate([values for values in value_arrays if values.size > 0])
    low = float(np.nanpercentile(merged, 0.5))
    high = float(np.nanpercentile(merged, 99.5))
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        return -1.0, 1.0
    padding = 0.05 * (high - low)
    return low - padding, high + padding


def weighted_hist(values: np.ndarray, weights: np.ndarray, bins: np.ndarray, normalize: bool) -> np.ndarray:
    hist = np.histogram(values, bins=bins, weights=weights)[0].astype(np.float64)
    if normalize:
        total = np.sum(hist)
        if total > 0.0:
            hist /= total
    return hist


def compare_summary(truth: np.ndarray, reco: np.ndarray) -> dict[str, float | int | None]:
    finite = np.isfinite(truth) & np.isfinite(reco)
    if not np.any(finite):
        return {
            "count": 0,
            "mean_diff": None,
            "mean_abs_diff": None,
            "rmse": None,
            "corr": None,
        }
    truth = truth[finite]
    reco = reco[finite]
    diff = reco - truth
    corr = None
    if truth.size >= 2:
        corr_matrix = np.corrcoef(truth, reco)
        corr_value = corr_matrix[0, 1]
        corr = float(corr_value) if np.isfinite(corr_value) else None
    return {
        "count": int(truth.size),
        "mean_diff": float(np.mean(diff)),
        "mean_abs_diff": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "corr": corr,
    }


def weighted_hist2d(x: np.ndarray, y: np.ndarray, weights: np.ndarray, bins: np.ndarray) -> np.ndarray:
    return np.histogram2d(x, y, bins=[bins, bins], weights=weights)[0].astype(np.float64)


def plot_chain_group(
    output_path: Path,
    title: str,
    variables: list[tuple[str, str]],
    target_chain: dict[str, np.ndarray],
    pred_chain: dict[str, np.ndarray],
    weights: np.ndarray,
) -> dict[str, dict[str, float | int | None]]:
    cols = 3
    rows = math.ceil(len(variables) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.9 * cols, 4.3 * rows), dpi=180)
    axes = np.asarray(axes).reshape(rows, cols)
    summary: dict[str, dict[str, float | int | None]] = {}
    for axis in axes.flat[len(variables):]:
        axis.axis("off")

    for axis, (label, base_name) in zip(axes.flat, variables):
        target_values = np.asarray(target_chain[base_name], dtype=np.float64)
        pred_values = np.asarray(pred_chain[base_name], dtype=np.float64)
        finite = np.isfinite(target_values) & np.isfinite(pred_values) & np.isfinite(weights) & (weights > 0.0)
        if not np.any(finite):
            axis.set_title(label)
            axis.text(0.5, 0.5, "No valid entries", ha="center", va="center", transform=axis.transAxes)
            axis.axis("off")
            summary[base_name] = compare_summary(np.array([], dtype=np.float64), np.array([], dtype=np.float64))
            continue
        x = target_values[finite]
        y = pred_values[finite]
        w = weights[finite]
        summary[base_name] = compare_summary(x, y)
        if base_name.endswith("_phi"):
            low, high = -np.pi, np.pi
        elif base_name.endswith("_eta"):
            bound = max(float(np.nanpercentile(np.abs(np.concatenate([x, y])), 99.5)), 1.0)
            low, high = -bound, bound
        elif base_name.endswith("_x") or base_name.endswith("_y") or base_name.endswith("_z") or base_name.endswith("_costheta"):
            low, high = -1.0, 1.0
        elif base_name.startswith("nu_") or base_name.startswith("tau_") or base_name == "mtautau":
            low = 0.0
            high = max(float(np.nanpercentile(np.concatenate([x, y]), 99.5)), 1.0)
        elif base_name.startswith("cos_theta_") or base_name == "theta_cm":
            low, high = observable_limits(base_name, x, y)
        else:
            low, high = observable_limits(base_name, x, y)
        bins = np.linspace(low, high, 60)
        hist2d = weighted_hist2d(x, y, w, bins)
        mesh = axis.pcolormesh(bins, bins, hist2d.T, cmap="Blues", shading="auto", vmin=0.0)
        fig.colorbar(mesh, ax=axis, fraction=0.046, pad=0.03, label="Entries")
        axis.plot([low, high], [low, high], color="black", linestyle="--", linewidth=1.0)
        axis.set_xlim(low, high)
        axis.set_ylim(low, high)
        axis.set_xlabel(f"Target {label}")
        axis.set_ylabel(f"Predicted {label}")
        axis.set_title(label)
        axis.grid(alpha=0.16)

    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return summary


def plot_observable(
    output_path: Path,
    observable: str,
    target_truth: np.ndarray,
    target_reco: np.ndarray,
    target_weights: np.ndarray,
    pred_truth: np.ndarray | None,
    pred_reco: np.ndarray | None,
    pred_weights: np.ndarray | None,
    pred_neutrino_truth: dict[str, np.ndarray] | None,
    pred_neutrino_pred: dict[str, np.ndarray] | None,
    pred_neutrino_weights: np.ndarray | None,
    normalize: bool,
) -> None:
    low, high = observable_limits(
        observable,
        target_truth,
        target_reco,
        pred_truth if pred_truth is not None else np.array([], dtype=np.float64),
        pred_reco if pred_reco is not None else np.array([], dtype=np.float64),
    )
    bins = np.linspace(low, high, 60)
    has_pred = pred_reco is not None and pred_reco.size > 0
    row_count = 2 if has_pred and pred_neutrino_truth is not None and pred_neutrino_pred is not None else 1
    col_count = 3 if has_pred else 2
    width_ratios = [1.1, 1.0, 1.0] if has_pred else [1.1, 1.0]
    fig, axes = plt.subplots(
        row_count,
        col_count,
        figsize=(15.2 if has_pred else 10.8, 8.8 if row_count == 2 else 4.6),
        dpi=180,
        gridspec_kw={"width_ratios": width_ratios},
    )
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes[np.newaxis, :]

    truth_hist = weighted_hist(target_truth, target_weights, bins, normalize)
    target_hist = weighted_hist(target_reco, target_weights, bins, normalize)
    axes[0, 0].step(bins[:-1], truth_hist, where="post", color="black", linewidth=1.7, label="Stored truth")
    axes[0, 0].step(bins[:-1], target_hist, where="post", color="#D55E00", linewidth=1.7, label="Target missing + visible")
    if has_pred:
        pred_hist = weighted_hist(pred_reco, pred_weights, bins, normalize)
        axes[0, 0].step(bins[:-1], pred_hist, where="post", color="#0072B2", linewidth=1.7, label="Predicted missing + visible")
    axes[0, 0].set_xlabel(observable)
    axes[0, 0].set_ylabel("Normalized yield" if normalize else "Weighted yield")
    axes[0, 0].set_title("1D overlay")
    axes[0, 0].grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=8)

    hist2d = weighted_hist2d(target_truth, target_reco, target_weights, bins)
    mesh = axes[0, 1].pcolormesh(bins, bins, hist2d.T, cmap="Blues", shading="auto", vmin=0.0)
    fig.colorbar(mesh, ax=axes[0, 1], fraction=0.046, pad=0.03, label="Entries")
    axes[0, 1].plot([low, high], [low, high], color="black", linestyle="--", linewidth=1.0)
    axes[0, 1].set_xlim(low, high)
    axes[0, 1].set_ylim(low, high)
    axes[0, 1].set_xlabel(f"Stored truth {observable}")
    axes[0, 1].set_ylabel(f"Target missing + visible {observable}")
    axes[0, 1].set_title("Truth vs target reco")
    axes[0, 1].grid(alpha=0.16)

    if has_pred:
        pred_hist2d = weighted_hist2d(pred_truth, pred_reco, pred_weights, bins)
        pred_mesh = axes[0, 2].pcolormesh(bins, bins, pred_hist2d.T, cmap="Blues", shading="auto", vmin=0.0)
        fig.colorbar(pred_mesh, ax=axes[0, 2], fraction=0.046, pad=0.03, label="Entries")
        axes[0, 2].plot([low, high], [low, high], color="black", linestyle="--", linewidth=1.0)
        axes[0, 2].set_xlim(low, high)
        axes[0, 2].set_ylim(low, high)
        axes[0, 2].set_xlabel(f"Stored truth {observable}")
        axes[0, 2].set_ylabel(f"Predicted missing + visible {observable}")
        axes[0, 2].set_title("Truth vs predicted reco")
        axes[0, 2].grid(alpha=0.16)

    if row_count == 2:
        for axis, component in zip(axes[1], ("pt", "eta", "phi")):
            truth_component = pred_neutrino_truth[component]
            pred_component = pred_neutrino_pred[component]
            if component == "phi":
                component_low, component_high = -np.pi, np.pi
            elif component == "eta":
                merged_component = np.concatenate([truth_component, pred_component])
                bound = max(float(np.nanpercentile(np.abs(merged_component), 99.5)), 1.0)
                component_low, component_high = -bound, bound
            else:
                merged_component = np.concatenate([truth_component, pred_component])
                component_low = 0.0
                component_high = max(float(np.nanpercentile(merged_component, 99.5)), 1.0)
            component_bins = np.linspace(component_low, component_high, 60)
            component_hist2d = weighted_hist2d(truth_component, pred_component, pred_neutrino_weights, component_bins)
            component_mesh = axis.pcolormesh(
                component_bins,
                component_bins,
                component_hist2d.T,
                cmap="Blues",
                shading="auto",
                vmin=0.0,
            )
            fig.colorbar(component_mesh, ax=axis, fraction=0.046, pad=0.03, label="Entries")
            axis.plot([component_low, component_high], [component_low, component_high], color="black", linestyle="--", linewidth=1.0)
            axis.set_xlim(component_low, component_high)
            axis.set_ylim(component_low, component_high)
            axis.set_xlabel(f"Truth neutrino {component}")
            axis.set_ylabel(f"Pred neutrino {component}")
            axis.set_title(f"Neutrino {component}")
            axis.grid(alpha=0.16)

    fig.suptitle(observable)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    parquet_paths = resolve_parquet_inputs(args.parquet)
    observables = requested_observables(args.observables)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    collected_target_truth: dict[str, list[np.ndarray]] = {observable: [] for observable in observables}
    collected_target_reco: dict[str, list[np.ndarray]] = {observable: [] for observable in observables}
    collected_target_weights: dict[str, list[np.ndarray]] = {observable: [] for observable in observables}
    collected_pred_truth: dict[str, list[np.ndarray]] = {observable: [] for observable in observables}
    collected_pred_reco: dict[str, list[np.ndarray]] = {observable: [] for observable in observables}
    collected_pred_weights: dict[str, list[np.ndarray]] = {observable: [] for observable in observables}
    collected_pred_neutrino_truth: dict[str, list[np.ndarray]] = {component: [] for component in ("pt", "eta", "phi")}
    collected_pred_neutrino_pred: dict[str, list[np.ndarray]] = {component: [] for component in ("pt", "eta", "phi")}
    collected_pred_neutrino_weights: list[np.ndarray] = []
    chain_variable_names = [
        "nu_a_pt", "nu_a_eta", "nu_a_phi",
        "nu_b_pt", "nu_b_eta", "nu_b_phi",
        "tau_a_pt", "tau_a_eta", "tau_a_phi", "tau_a_energy",
        "tau_b_pt", "tau_b_eta", "tau_b_phi", "tau_b_energy",
        "tau_a_mass", "tau_b_mass",
        "tau_a_cm_costheta", "tau_b_cm_costheta",
        "tau_a_cm_phi", "tau_b_cm_phi",
        "vis_a_rest_x", "vis_a_rest_y", "vis_a_rest_z",
        "vis_b_rest_x", "vis_b_rest_y", "vis_b_rest_z",
        "tautau_px", "tautau_py", "tautau_pz",
        "basis_a_rest_n_x", "basis_a_rest_n_y", "basis_a_rest_n_z",
        "basis_a_rest_r_x", "basis_a_rest_r_y", "basis_a_rest_r_z",
        "basis_a_rest_k_x", "basis_a_rest_k_y", "basis_a_rest_k_z",
        "basis_b_rest_n_x", "basis_b_rest_n_y", "basis_b_rest_n_z",
        "basis_b_rest_r_x", "basis_b_rest_r_y", "basis_b_rest_r_z",
        "basis_b_rest_k_x", "basis_b_rest_k_y", "basis_b_rest_k_z",
        "theta_cm", "mtautau",
        "cos_theta_A_n", "cos_theta_A_r", "cos_theta_A_k",
        "cos_theta_B_n", "cos_theta_B_r", "cos_theta_B_k",
    ]
    collected_target_chain: dict[str, list[np.ndarray]] = {name: [] for name in chain_variable_names}
    collected_pred_chain: dict[str, list[np.ndarray]] = {name: [] for name in chain_variable_names}
    collected_chain_weights: list[np.ndarray] = []
    rows_seen = 0
    rows_used = 0

    remaining = args.max_entries
    for path_index, path in enumerate(parquet_paths, start=1):
        print(f"[check-target-missing-vs-truth] loading {path_index}/{len(parquet_paths)} path={path}", flush=True)
        path_limit = None if remaining is None else remaining
        for batch in iter_event_batches(
            path,
            observables,
            args.batch_size,
            path_limit,
            args.region,
            args.truth_region_only,
        ):
            rows_seen += len(batch)
            selected = apply_event_selection(batch, args.region, args.truth_region_only)
            if len(selected) == 0:
                continue
            rows_used += len(selected)
            weights = event_weights(selected, args.weight_field)
            if args.debug_chain:
                try:
                    target_chain_batch = reconstructed_chain_values(selected, "target")
                    pred_chain_batch = reconstructed_chain_values(selected, "pred")
                    chain_finite = np.isfinite(weights) & (weights > 0.0)
                    for name in chain_variable_names:
                        chain_finite &= np.isfinite(target_chain_batch[f"target_{name}"])
                        chain_finite &= np.isfinite(pred_chain_batch[f"pred_{name}"])
                    if np.any(chain_finite):
                        for name in chain_variable_names:
                            collected_target_chain[name].append(target_chain_batch[f"target_{name}"][chain_finite])
                            collected_pred_chain[name].append(pred_chain_batch[f"pred_{name}"][chain_finite])
                        collected_chain_weights.append(weights[chain_finite])
                except Exception as e:
                    print(e)
                    pass
            try:
                pred_neutrino_truth_batch = {
                    component: flattened_leg_feature(selected, "target_invisible", component)
                    for component in ("pt", "eta", "phi")
                }
                pred_neutrino_pred_batch = {
                    component: flattened_leg_feature(selected, "pred_invisible", component)
                    for component in ("pt", "eta", "phi")
                }
                neutrino_weights_batch = np.repeat(weights, 2)
                neutrino_finite = (
                    np.isfinite(neutrino_weights_batch)
                    & (neutrino_weights_batch > 0.0)
                    & np.isfinite(pred_neutrino_truth_batch["pt"])
                    & np.isfinite(pred_neutrino_truth_batch["eta"])
                    & np.isfinite(pred_neutrino_truth_batch["phi"])
                    & np.isfinite(pred_neutrino_pred_batch["pt"])
                    & np.isfinite(pred_neutrino_pred_batch["eta"])
                    & np.isfinite(pred_neutrino_pred_batch["phi"])
                )
                if np.any(neutrino_finite):
                    for component in ("pt", "eta", "phi"):
                        collected_pred_neutrino_truth[component].append(pred_neutrino_truth_batch[component][neutrino_finite])
                        collected_pred_neutrino_pred[component].append(pred_neutrino_pred_batch[component][neutrino_finite])
                    collected_pred_neutrino_weights.append(neutrino_weights_batch[neutrino_finite])
            except Exception as e:
                print(e)
                pass
            for observable in observables:
                truth_field = f"truth_{observable}"
                if truth_field not in selected.fields:
                    continue
                truth = truth_values(selected, observable)
                target_reco = target_reconstructed_values(selected, observable)
                try:
                    pred_reco = predicted_reconstructed_values(selected, observable)
                except Exception as e:
                    print(e, "pred reco None")
                    pred_reco = None

                finite_target = np.isfinite(truth) & np.isfinite(target_reco) & np.isfinite(weights) & (weights > 0.0)
                if np.any(finite_target):
                    collected_target_truth[observable].append(truth[finite_target])
                    collected_target_reco[observable].append(target_reco[finite_target])
                    collected_target_weights[observable].append(weights[finite_target])
                if pred_reco is not None:
                    finite_pred = np.isfinite(truth) & np.isfinite(pred_reco) & np.isfinite(weights) & (weights > 0.0)
                    if np.any(finite_pred):
                        collected_pred_truth[observable].append(truth[finite_pred])
                        collected_pred_reco[observable].append(pred_reco[finite_pred])
                        collected_pred_weights[observable].append(weights[finite_pred])
                if not np.any(finite_target) and pred_reco is None:
                    continue
            if remaining is not None:
                remaining -= len(batch)
                if remaining <= 0:
                    break
        if remaining is not None and remaining <= 0:
            break

    summary: dict[str, Any] = {
        "inputs": [str(path) for path in parquet_paths],
        "rows_seen": rows_seen,
        "rows_used": rows_used,
        "region": args.region,
        "truth_region_only": args.truth_region_only,
        "weight_field": args.weight_field,
        "observables": {},
    }

    if args.debug_chain and collected_chain_weights:
        target_chain = {
            name: np.concatenate(chunks) if chunks else np.array([], dtype=np.float64)
            for name, chunks in collected_target_chain.items()
        }
        pred_chain = {
            name: np.concatenate(chunks) if chunks else np.array([], dtype=np.float64)
            for name, chunks in collected_pred_chain.items()
        }
        chain_weights = np.concatenate(collected_chain_weights)
        chain_output_dir = output_dir / "debug_chain"
        chain_summary = {
            "neutrino": plot_chain_group(
                chain_output_dir / "missing_neutrino_target_vs_pred.png",
                "Missing neutrino: Target vs Predicted",
                [
                    ("nu_a_pt", "nu_a_pt"),
                    ("nu_a_eta", "nu_a_eta"),
                    ("nu_a_phi", "nu_a_phi"),
                    ("nu_b_pt", "nu_b_pt"),
                    ("nu_b_eta", "nu_b_eta"),
                    ("nu_b_phi", "nu_b_phi"),
                ],
                target_chain,
                pred_chain,
                chain_weights,
            ),
            "tau": plot_chain_group(
                chain_output_dir / "tau_projected_target_vs_pred.png",
                "Tau after mass projection: Target vs Predicted",
                [
                    ("tau_a_pt", "tau_a_pt"),
                    ("tau_a_eta", "tau_a_eta"),
                    ("tau_a_phi", "tau_a_phi"),
                    ("tau_a_energy", "tau_a_energy"),
                    ("tau_b_pt", "tau_b_pt"),
                    ("tau_b_eta", "tau_b_eta"),
                    ("tau_b_phi", "tau_b_phi"),
                    ("tau_b_energy", "tau_b_energy")
                ],
                target_chain,
                pred_chain,
                chain_weights,
            ),
            "cm_frame": plot_chain_group(
                chain_output_dir / "cm_frame_target_vs_pred.png",
                "CM-frame kinematics: Target vs Predicted",
                [
                    ("tau_a_cm_costheta", "tau_a_cm_costheta"),
                    ("tau_b_cm_costheta", "tau_b_cm_costheta"),
                    ("tau_a_cm_phi", "tau_a_cm_phi"),
                    ("tau_b_cm_phi", "tau_b_cm_phi"),
                ],
                target_chain,
                pred_chain,
                chain_weights,
            ),
            "tautau": plot_chain_group(
                chain_output_dir / "tautau_target_vs_pred.png",
                "tautau: Target vs Predicted",
                [
                    ("tautau_px", "tautau_px"),
                    ("tautau_py", "tautau_py"),
                    ("tautau_pz", "tautau_pz"),

                ],
                target_chain,
                pred_chain,
                chain_weights,
            ),
            "visible_rest": plot_chain_group(
                chain_output_dir / "visible_rest_unit_target_vs_pred.png",
                "Visible direction in tau rest frame: Target vs Predicted",
                [
                    ("vis_a_rest_x", "vis_a_rest_x"),
                    ("vis_a_rest_y", "vis_a_rest_y"),
                    ("vis_a_rest_z", "vis_a_rest_z"),
                    ("vis_b_rest_x", "vis_b_rest_x"),
                    ("vis_b_rest_y", "vis_b_rest_y"),
                    ("vis_b_rest_z", "vis_b_rest_z"),
                ],
                target_chain,
                pred_chain,
                chain_weights,
            ),
            "basis_a_rest": plot_chain_group(
                chain_output_dir / "basis_a_rest_target_vs_pred.png",
                "Helicity basis in tau-A rest frame: Target vs Predicted",
                [
                    ("basis_a_rest_n_x", "basis_a_rest_n_x"),
                    ("basis_a_rest_n_y", "basis_a_rest_n_y"),
                    ("basis_a_rest_n_z", "basis_a_rest_n_z"),
                    ("basis_a_rest_r_x", "basis_a_rest_r_x"),
                    ("basis_a_rest_r_y", "basis_a_rest_r_y"),
                    ("basis_a_rest_r_z", "basis_a_rest_r_z"),
                    ("basis_a_rest_k_x", "basis_a_rest_k_x"),
                    ("basis_a_rest_k_y", "basis_a_rest_k_y"),
                    ("basis_a_rest_k_z", "basis_a_rest_k_z"),
                ],
                target_chain,
                pred_chain,
                chain_weights,
            ),
            "basis_b_rest": plot_chain_group(
                chain_output_dir / "basis_b_rest_target_vs_pred.png",
                "Helicity basis in tau-B rest frame: Target vs Predicted",
                [
                    ("basis_b_rest_n_x", "basis_b_rest_n_x"),
                    ("basis_b_rest_n_y", "basis_b_rest_n_y"),
                    ("basis_b_rest_n_z", "basis_b_rest_n_z"),
                    ("basis_b_rest_r_x", "basis_b_rest_r_x"),
                    ("basis_b_rest_r_y", "basis_b_rest_r_y"),
                    ("basis_b_rest_r_z", "basis_b_rest_r_z"),
                    ("basis_b_rest_k_x", "basis_b_rest_k_x"),
                    ("basis_b_rest_k_y", "basis_b_rest_k_y"),
                    ("basis_b_rest_k_z", "basis_b_rest_k_z"),
                ],
                target_chain,
                pred_chain,
                chain_weights,
            ),
            "global": plot_chain_group(
                chain_output_dir / "global_observables_target_vs_pred.png",
                "Global observables: Target vs Predicted",
                [
                    ("tau_a_mass", "tau_a_mass"),
                    ("tau_b_mass", "tau_b_mass"),
                    ("theta_cm", "theta_cm"),
                    ("mtautau", "mtautau"),
                ],
                target_chain,
                pred_chain,
                chain_weights,
            ),
            "angular": plot_chain_group(
                chain_output_dir / "angular_observables_target_vs_pred.png",
                "Angular observables: Target vs Predicted",
                [
                    ("cos_theta_A_n", "cos_theta_A_n"),
                    ("cos_theta_A_r", "cos_theta_A_r"),
                    ("cos_theta_A_k", "cos_theta_A_k"),
                    ("cos_theta_B_n", "cos_theta_B_n"),
                    ("cos_theta_B_r", "cos_theta_B_r"),
                    ("cos_theta_B_k", "cos_theta_B_k"),
                ],
                target_chain,
                pred_chain,
                chain_weights,
            ),
        }
        summary["debug_chain"] = {
            "plots": {
                "neutrino": str((chain_output_dir / "missing_neutrino_target_vs_pred.png").relative_to(output_dir)),
                "tau": str((chain_output_dir / "tau_projected_target_vs_pred.png").relative_to(output_dir)),
                "cm_frame": str((chain_output_dir / "cm_frame_target_vs_pred.png").relative_to(output_dir)),
                "visible_rest": str((chain_output_dir / "visible_rest_unit_target_vs_pred.png").relative_to(output_dir)),
                "basis_a_rest": str((chain_output_dir / "basis_a_rest_target_vs_pred.png").relative_to(output_dir)),
                "basis_b_rest": str((chain_output_dir / "basis_b_rest_target_vs_pred.png").relative_to(output_dir)),
                "global": str((chain_output_dir / "global_observables_target_vs_pred.png").relative_to(output_dir)),
                "angular": str((chain_output_dir / "angular_observables_target_vs_pred.png").relative_to(output_dir)),
            },
            "summary": chain_summary,
        }

    for observable in observables:
        if not collected_target_truth[observable]:
            continue
        target_truth = np.concatenate(collected_target_truth[observable])
        target_reco = np.concatenate(collected_target_reco[observable])
        target_weights = np.concatenate(collected_target_weights[observable])
        observable_summary: dict[str, Any] = {
            "target_vs_truth": compare_summary(target_truth, target_reco),
        }
        pred_truth = None
        pred_reco = None
        pred_weights = None
        pred_neutrino_truth = {
            component: np.concatenate(chunks) if chunks else np.array([], dtype=np.float64)
            for component, chunks in collected_pred_neutrino_truth.items()
        }
        pred_neutrino_pred = {
            component: np.concatenate(chunks) if chunks else np.array([], dtype=np.float64)
            for component, chunks in collected_pred_neutrino_pred.items()
        }
        pred_neutrino_weights = (
            np.concatenate(collected_pred_neutrino_weights)
            if collected_pred_neutrino_weights
            else np.array([], dtype=np.float64)
        )
        if pred_neutrino_weights.size == 0:
            pred_neutrino_truth = None
            pred_neutrino_pred = None
            pred_neutrino_weights = None
        if collected_pred_reco[observable]:
            pred_truth = np.concatenate(collected_pred_truth[observable])
            pred_reco = np.concatenate(collected_pred_reco[observable])
            pred_weights = np.concatenate(collected_pred_weights[observable])
            observable_summary["predicted_vs_truth"] = compare_summary(pred_truth, pred_reco)
        summary["observables"][observable] = observable_summary
        plot_path = output_dir / "plots" / f"{sanitize_filename(observable)}.png"
        plot_observable(
            plot_path,
            observable,
            target_truth,
            target_reco,
            target_weights,
            pred_truth,
            pred_reco,
            pred_weights,
            pred_neutrino_truth,
            pred_neutrino_pred,
            pred_neutrino_weights,
            args.normalize,
        )
        summary["observables"][observable]["plot"] = str(plot_path.relative_to(output_dir))
        print(f"[check-target-missing-vs-truth] wrote observable={observable} plot={plot_path}", flush=True)

    summary_path = output_dir / "summary.json"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"[check-target-missing-vs-truth] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
