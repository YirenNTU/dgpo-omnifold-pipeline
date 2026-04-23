from __future__ import annotations

import awkward as ak
import numpy as np
import vector

PHOTON_DR_MAX = 0.3
MAX_PART_ENERGY_GEV = 91.25
FOUR_VECTOR_FEATURES = ["energy", "pt", "eta", "phi"]
VISIBLE_KIND_PRIORITY = {
    "electron": 0,
    "muon": 1,
    "pion": 2,
    "rho": 3,
    "other": 4,
}
DEFAULT_PART_AUX_FIELDS = [
    "Part_charge",
    "Part_pdgId",
    "Part_vtxIdx",
    "Part_hpcShowerEnergy",
    "Part_hpcShowerTheta",
    "Part_hpcShowerPhi",
    "Part_hpcParticleCode",
    "Part_hpcNumLayers",
    "Part_hpcLayerHitPattern",
    "Part_hpcNumAssociatedShowers",
    "Part_hpcTotalShowerEnergy",
    "Part_hacShowerEnergy",
    "Part_hacShowerTheta",
    "Part_hacShowerPhi",
    "Part_hacParticleCode",
    "Part_hacNumTowers",
    "Part_hacTowerHitPattern",
    "Part_hacNumAssociatedShowers",
    "Part_hacTotalShowerEnergy",
    "Part_sticShowerEnergy",
    "Part_sticShowerTheta",
    "Part_sticShowerPhi",
    "Part_sticNumTowers",
    "Part_sticChargedTag",
    "Part_sticSiliconVertexPos",
    "Part_hemisphere",
]
DEFAULT_GLOBAL_FIELDS = [
    "Event_totalChargedEnergy",
    "Event_totalEMEnergy",
    "Event_totalHadronicEnergy",
    "thrust_Mag",
    "charged_E",
    "missing_px",
    "missing_py",
    "missing_pt",
    "isolation_angle",
    "thrust_x",
    "thrust_y",
    "thrust_z",
]
vector.register_awkward()


def build_momentum4d(px, py, pz, energy):
    return ak.zip(
        {
            "px": px,
            "py": py,
            "pz": pz,
            "E": energy,
        },
        with_name="Momentum4D",
    )


def build_part_momentum4d(events: ak.Array):
    return build_momentum4d(
        px = events["Part_fourMomentum_fCoordinates_fX"],
        py = events["Part_fourMomentum_fCoordinates_fY"],
        pz = events["Part_fourMomentum_fCoordinates_fZ"],
        energy = events["Part_fourMomentum_fCoordinates_fT"],
    )


def part_energy_mask(events: ak.Array) -> ak.Array:
    if "Part_fourMomentum_fCoordinates_fT" not in events.fields:
        return ak.ones_like(events["Part_pdgId"], dtype=bool)

    energy = events["Part_fourMomentum_fCoordinates_fT"]
    return ak.values_astype(np.isfinite(energy) & (energy <= MAX_PART_ENERGY_GEV), bool)


def filter_part_values(events: ak.Array, values: ak.Array) -> ak.Array:
    return values[part_energy_mask(events)]


def build_filtered_part_momentum4d(events: ak.Array):
    return filter_part_values(events, build_part_momentum4d(events))


def compute_pt_eta_phi(px: ak.Array, py: ak.Array, pz: ak.Array) -> tuple[ak.Array, ak.Array, ak.Array]:
    p4 = build_momentum4d(px, py, pz, np.sqrt(px * px + py * py + pz * pz))
    return p4.pt, p4.eta, p4.phi


def features_from_p4(p4, feature_names=FOUR_VECTOR_FEATURES):
    if isinstance(p4, ak.Array):
        components = []
        for feature_name in feature_names:
            if feature_name in {"energy", "E"}:
                values = p4.E
            elif feature_name == "px":
                values = p4.px
            elif feature_name == "py":
                values = p4.py
            elif feature_name == "pz":
                values = p4.pz
            elif feature_name == "mass":
                values = ak.where(np.isfinite(p4.mass), p4.mass, 0.0)
            elif feature_name == "pt":
                values = p4.pt
            elif feature_name == "eta":
                values = ak.where(np.isfinite(p4.eta), p4.eta, 0.0)
            elif feature_name == "phi":
                values = ak.where(np.isfinite(p4.phi), p4.phi, 0.0)
            else:
                raise ValueError(f"Unsupported four-vector feature '{feature_name}'.")
            components.append(ak.values_astype(values, np.float32)[..., np.newaxis])
        return ak.concatenate(components, axis=-1)

    components = []
    for feature_name in feature_names:
        if feature_name in {"energy", "E"}:
            values = np.asarray(p4.E, dtype=np.float32)
        elif feature_name == "px":
            values = np.asarray(p4.px, dtype=np.float32)
        elif feature_name == "py":
            values = np.asarray(p4.py, dtype=np.float32)
        elif feature_name == "pz":
            values = np.asarray(p4.pz, dtype=np.float32)
        elif feature_name == "mass":
            values = np.asarray(p4.mass, dtype=np.float32)
            values = np.where(np.isfinite(values), values, 0.0)
        elif feature_name == "pt":
            values = np.asarray(p4.pt, dtype=np.float32)
        elif feature_name == "eta":
            values = np.asarray(p4.eta, dtype=np.float32)
            values = np.where(np.isfinite(values), values, 0.0)
        elif feature_name == "phi":
            values = np.asarray(p4.phi, dtype=np.float32)
            values = np.where(np.isfinite(values), values, 0.0)
        else:
            raise ValueError(f"Unsupported four-vector feature '{feature_name}'.")
        components.append(values)
    return np.stack(components, axis=-1).astype(np.float32)


def stack_tau_pair(tau_minus, tau_plus):
    return ak.concatenate([tau_minus[:, np.newaxis], tau_plus[:, np.newaxis]], axis=1)


def stack_tau_pair_mask(tau_minus_mask, tau_plus_mask):
    return ak.concatenate([tau_minus_mask[:, np.newaxis], tau_plus_mask[:, np.newaxis]], axis=1)


def zero_p4(shape):
    zeros = np.zeros(shape, dtype=np.float32)
    return build_momentum4d(zeros, zeros, zeros, zeros)


def mask_p4(p4, mask):
    return build_momentum4d(
        ak.where(mask, p4.px, 0.0),
        ak.where(mask, p4.py, 0.0),
        ak.where(mask, p4.pz, 0.0),
        ak.where(mask, p4.E, 0.0),
    )


def sum_masked_p4(events: ak.Array, mask: ak.Array):
    part_p4 = build_part_momentum4d(events)
    return ak.sum(part_p4[mask & part_energy_mask(events)], axis=1)


def map_hemisphere_to_tau_sign(first_values, second_values, first_charge, second_charge):
    first_is_tau_minus = (first_charge < 0) & (second_charge > 0)
    second_is_tau_minus = (second_charge < 0) & (first_charge > 0)
    tau_minus_mask = first_is_tau_minus | second_is_tau_minus
    tau_plus_mask = tau_minus_mask

    tau_minus = ak.where(first_is_tau_minus, first_values, second_values)
    tau_plus = ak.where(first_is_tau_minus, second_values, first_values)
    return tau_minus, tau_plus, tau_minus_mask, tau_plus_mask


def reorder_tau_pair(pair_values, swap_mask):
    first = ak.where(swap_mask, pair_values[:, 1], pair_values[:, 0])
    second = ak.where(swap_mask, pair_values[:, 0], pair_values[:, 1])
    return ak.concatenate([first[:, np.newaxis], second[:, np.newaxis]], axis=1)


def _classify_visible_kind_rank(lead_abs_pdg, has_nearby_photon):
    return ak.where(
        lead_abs_pdg == 2,
        VISIBLE_KIND_PRIORITY["electron"],
        ak.where(
            lead_abs_pdg == 6,
            VISIBLE_KIND_PRIORITY["muon"],
            ak.where(
                lead_abs_pdg == 41,
                ak.where(
                    has_nearby_photon,
                    VISIBLE_KIND_PRIORITY["rho"],
                    VISIBLE_KIND_PRIORITY["pion"],
                ),
                VISIBLE_KIND_PRIORITY["other"],
            ),
        ),
    )


def _build_hemisphere_masks(events: ak.Array, part_p4: ak.Array, charge: ak.Array, energy_valid: ak.Array):
    if "Part_in_hemisphere_a" in events.fields and "Part_in_hemisphere_b" in events.fields:
        return {
            "a": ak.values_astype(events["Part_in_hemisphere_a"], bool) & energy_valid,
            "b": ak.values_astype(events["Part_in_hemisphere_b"], bool) & energy_valid,
        }

    if "Part_hemisphere" in events.fields:
        hemisphere = events["Part_hemisphere"]
        return {
            "a": (hemisphere == 1) & energy_valid,
            "b": (hemisphere == -1) & energy_valid,
        }

    required_thrust = {"thrust_x", "thrust_y", "thrust_z"}
    if not required_thrust.issubset(set(events.fields)):
        raise ValueError(
            "Visible tau construction needs either Part_in_hemisphere_a/b, Part_hemisphere, "
            "or thrust_x/y/z to infer hemispheres."
        )

    thrust = vector.zip({"x": events["thrust_x"], "y": events["thrust_y"], "z": events["thrust_z"]})
    raw_hemisphere = ak.values_astype(part_p4.to_Vector3D().dot(thrust) > 0, bool)
    part_idx_all = ak.local_index(events["Part_pdgId"])

    lead_charges = {}
    lead_flags = {}
    raw_masks = {"a": raw_hemisphere, "b": ~raw_hemisphere}
    for hemisphere_name, raw_mask in raw_masks.items():
        charged_mask = raw_mask & (charge != 0) & energy_valid
        charged_p4 = part_p4[charged_mask]
        charged_idx = part_idx_all[charged_mask]
        sorted_idx = ak.argsort(charged_p4.p, axis=1, ascending=False)
        leading_idx = ak.firsts(charged_idx[sorted_idx])
        lead_flag = part_idx_all == leading_idx
        lead_flags[hemisphere_name] = lead_flag
        lead_charges[hemisphere_name] = ak.fill_none(ak.firsts(charge[lead_flag]), 0)

    is_leading_os = (
        (lead_charges["a"] + lead_charges["b"] == 0)
        & (lead_charges["a"] != 0)
        & (lead_charges["b"] != 0)
    )
    switch_hemisphere = is_leading_os & (lead_charges["a"] < 0) & (lead_charges["b"] > 0)
    switch_broadcasted = ak.broadcast_arrays(switch_hemisphere, raw_hemisphere)[0]
    return {
        "a": (raw_hemisphere ^ switch_broadcasted) & energy_valid,
        "b": ((~raw_hemisphere) ^ switch_broadcasted) & energy_valid,
    }


def _build_visible_tau_layout(events: ak.Array, include_nearby_photons: bool = True):
    charge = events["Part_charge"]
    pdg_id = abs(events["Part_pdgId"])
    part_p4 = build_part_momentum4d(events)
    energy_valid = part_energy_mask(events)

    hemisphere_masks = _build_hemisphere_masks(events, part_p4, charge, energy_valid)

    prong_charge_sums = {}
    visible_p4_by_hemisphere = {}
    visible_kind_rank_by_hemisphere = {}

    for hemisphere_name, hemisphere_mask in hemisphere_masks.items():
        prong_mask = hemisphere_mask & (charge != 0)
        photon_mask = hemisphere_mask & (charge == 0) & (pdg_id == 21)
        prong_p4_constituents = part_p4[prong_mask]
        photon_p4_constituents = part_p4[photon_mask]
        prong_abs_pdg = pdg_id[prong_mask]

        prong_p4 = sum_masked_p4(events, prong_mask)
        prong_charge_sums[hemisphere_name] = ak.values_astype(ak.sum(charge[prong_mask], axis=1), np.float32)

        prong_sort = ak.argsort(prong_p4_constituents.p, axis=1, ascending=False)
        lead_prong_abs_pdg = ak.values_astype(
            ak.fill_none(ak.firsts(prong_abs_pdg[prong_sort]), 0),
            np.int64,
        )

        pairs = ak.cartesian(
            {
                "photon": photon_p4_constituents,
                "prong": prong_p4_constituents,
            },
            axis=1,
            nested=True,
        )
        delta_r = pairs["photon"].deltaR(pairs["prong"])
        photon_near_prong = ak.fill_none(ak.any(delta_r < PHOTON_DR_MAX, axis=-1), False)
        has_nearby_photon = ak.fill_none(ak.any(photon_near_prong, axis=1), False)

        photon_p4 = ak.sum(photon_p4_constituents[photon_near_prong], axis=1)
        visible_p4_by_hemisphere[hemisphere_name] = prong_p4 + (
            photon_p4 if include_nearby_photons else zero_p4((len(events),))
        )
        visible_kind_rank_by_hemisphere[hemisphere_name] = _classify_visible_kind_rank(
            lead_prong_abs_pdg,
            has_nearby_photon,
        )

    visible_tau_minus, visible_tau_plus, visible_tau_minus_mask, visible_tau_plus_mask = map_hemisphere_to_tau_sign(
        visible_p4_by_hemisphere["a"],
        visible_p4_by_hemisphere["b"],
        prong_charge_sums["a"],
        prong_charge_sums["b"],
    )
    visible_kind_rank_minus, visible_kind_rank_plus, _, _ = map_hemisphere_to_tau_sign(
        visible_kind_rank_by_hemisphere["a"],
        visible_kind_rank_by_hemisphere["b"],
        prong_charge_sums["a"],
        prong_charge_sums["b"],
    )
    visible_charge_minus, visible_charge_plus, _, _ = map_hemisphere_to_tau_sign(
        prong_charge_sums["a"],
        prong_charge_sums["b"],
        prong_charge_sums["a"],
        prong_charge_sums["b"],
    )

    visible_tau = stack_tau_pair(visible_tau_minus, visible_tau_plus)
    visible_tau_mask = stack_tau_pair_mask(visible_tau_minus_mask, visible_tau_plus_mask)
    visible_kind_rank = stack_tau_pair(visible_kind_rank_minus, visible_kind_rank_plus)
    visible_charge = stack_tau_pair(visible_charge_minus, visible_charge_plus)

    visible_kind_rank_0 = ak.to_numpy(visible_kind_rank[:, 0], allow_missing=False)
    visible_kind_rank_1 = ak.to_numpy(visible_kind_rank[:, 1], allow_missing=False)
    visible_charge_0 = ak.to_numpy(visible_charge[:, 0], allow_missing=False)
    visible_charge_1 = ak.to_numpy(visible_charge[:, 1], allow_missing=False)
    visible_mask_0 = ak.to_numpy(visible_tau_mask[:, 0], allow_missing=False)
    visible_mask_1 = ak.to_numpy(visible_tau_mask[:, 1], allow_missing=False)

    swap_mask = (~visible_mask_0 & visible_mask_1) | (
        visible_mask_0 == visible_mask_1
    ) & (
        (visible_kind_rank_0 > visible_kind_rank_1)
        | (
            (visible_kind_rank_0 == visible_kind_rank_1)
            & (visible_charge_0 < visible_charge_1)
        )
    )
    swap_mask = ak.Array(swap_mask)

    return reorder_tau_pair(visible_tau, swap_mask), reorder_tau_pair(visible_tau_mask, swap_mask), swap_mask


def build_visible_tau_assumption_p4(events: ak.Array):
    visible_tau, visible_tau_mask, _ = _build_visible_tau_layout(events)

    # Keep the legacy dual return structure to avoid changing downstream consumers.
    return (
        visible_tau,
        visible_tau_mask,
        visible_tau,
        visible_tau_mask,
    )


def build_visible_tau_assumptions(events: ak.Array):
    # Shape convention is [event, visible_slot(2)] in canonical visible-type order.
    return build_visible_tau_assumption_p4(events)


def build_prong_only_visible_tau_p4(events: ak.Array):
    return _build_visible_tau_layout(events, include_nearby_photons=False)


def truth_feature(values: ak.Array | None):
    if values is None:
        return ak.Array([])
    return ak.values_astype(ak.fill_none(values, np.nan), np.float32)


def build_tau_targets(
    events: ak.Array,
    tau_vis_prong_p4,
    tau_vis_prong_mask,
    tau_vis_rho_p4,
    tau_vis_rho_mask,
    slot_swap_mask=None,
):
    n_events = len(events)
    x_invisible_p4 = zero_p4((n_events, 2))
    x_invisible_mask = ak.Array(np.zeros((n_events, 2), dtype=bool))
    num_invisible_raw = np.zeros(n_events, dtype=np.int64)
    num_invisible_valid = np.zeros(n_events, dtype=np.int64)
    tau_vis_target_p4 = tau_vis_prong_p4
    tau_vis_target_mask = tau_vis_prong_mask

    required_fields = [
        "truth_tau_px",
        "truth_tau_py",
        "truth_tau_pz",
        "truth_tau_E",
        "truth_anti_tau_px",
        "truth_anti_tau_py",
        "truth_anti_tau_pz",
        "truth_anti_tau_E",
    ]
    if not all(field in events.fields for field in required_fields):
        return x_invisible_p4, x_invisible_mask, num_invisible_raw, num_invisible_valid, tau_vis_target_p4, tau_vis_target_mask

    truth_tau = build_momentum4d(
        truth_feature(events["truth_tau_px"]),
        truth_feature(events["truth_tau_py"]),
        truth_feature(events["truth_tau_pz"]),
        truth_feature(events["truth_tau_E"]),
    )
    truth_anti_tau = build_momentum4d(
        truth_feature(events["truth_anti_tau_px"]),
        truth_feature(events["truth_anti_tau_py"]),
        truth_feature(events["truth_anti_tau_pz"]),
        truth_feature(events["truth_anti_tau_E"]),
    )
    tau_truth_valid = np.isfinite(truth_tau.E) & np.isfinite(truth_tau.px) & np.isfinite(truth_tau.py) & np.isfinite(truth_tau.pz)
    anti_tau_truth_valid = np.isfinite(truth_anti_tau.E) & np.isfinite(truth_anti_tau.px) & np.isfinite(truth_anti_tau.py) & np.isfinite(truth_anti_tau.pz)
    truth_tau_pair = stack_tau_pair(truth_tau, truth_anti_tau)
    truth_tau_valid_pair = stack_tau_pair_mask(tau_truth_valid, anti_tau_truth_valid)

    if slot_swap_mask is None:
        _, _, slot_swap_mask = _build_visible_tau_layout(events)

    truth_tau_pair = reorder_tau_pair(truth_tau_pair, slot_swap_mask)
    truth_tau_valid_pair = reorder_tau_pair(truth_tau_valid_pair, slot_swap_mask)
    x_invisible_p4 = truth_tau_pair - tau_vis_target_p4
    x_invisible_mask = truth_tau_valid_pair & tau_vis_target_mask
    x_invisible_p4 = mask_p4(x_invisible_p4, x_invisible_mask)
    tau_vis_target_p4 = mask_p4(tau_vis_target_p4, tau_vis_target_mask)

    num_invisible_raw[:] = 2
    num_invisible_valid[:] = ak.to_numpy(ak.sum(ak.values_astype(x_invisible_mask, np.int64), axis=1), allow_missing=False).astype(np.int64)
    return x_invisible_p4, x_invisible_mask, num_invisible_raw, num_invisible_valid, tau_vis_target_p4, tau_vis_target_mask


def extract_target_invisible_observable(events: ak.Array, observable: str) -> np.ndarray:
    tau_vis_prong_p4, tau_vis_prong_mask, tau_vis_rho_p4, tau_vis_rho_mask = build_visible_tau_assumptions(events)
    x_invisible_p4, x_invisible_mask, _, _, _, _ = build_tau_targets(
        events,
        tau_vis_prong_p4,
        tau_vis_prong_mask,
        tau_vis_rho_p4,
        tau_vis_rho_mask,
    )

    if observable in {"energy", "E"}:
        return ak.to_numpy(ak.flatten(x_invisible_p4.E[x_invisible_mask], axis=None), allow_missing=False).astype(np.float32)
    if observable == "px":
        return ak.to_numpy(ak.flatten(x_invisible_p4.px[x_invisible_mask], axis=None), allow_missing=False).astype(np.float32)
    if observable == "py":
        return ak.to_numpy(ak.flatten(x_invisible_p4.py[x_invisible_mask], axis=None), allow_missing=False).astype(np.float32)
    if observable == "pz":
        return ak.to_numpy(ak.flatten(x_invisible_p4.pz[x_invisible_mask], axis=None), allow_missing=False).astype(np.float32)
    if observable == "pt":
        return ak.to_numpy(ak.flatten(x_invisible_p4.pt[x_invisible_mask], axis=None), allow_missing=False).astype(np.float32)
    if observable == "eta":
        return ak.to_numpy(ak.flatten(x_invisible_p4.eta[x_invisible_mask], axis=None), allow_missing=False).astype(np.float32)
    if observable == "phi":
        return ak.to_numpy(ak.flatten(x_invisible_p4.phi[x_invisible_mask], axis=None), allow_missing=False).astype(np.float32)
    if observable == "mass":
        return ak.to_numpy(ak.flatten(x_invisible_p4.mass[x_invisible_mask], axis=None), allow_missing=False).astype(np.float32)
    raise ValueError(f"Unsupported target invisible observable '{observable}'.")


def extract_visible_tau_observable(events: ak.Array, mode: str, observable: str) -> np.ndarray:
    tau_vis_prong_p4, tau_vis_prong_mask, tau_vis_rho_p4, tau_vis_rho_mask = build_visible_tau_assumptions(events)
    tau_vis_prong_only_p4, tau_vis_prong_only_mask, _ = build_prong_only_visible_tau_p4(events)

    if mode == "prong":
        values = tau_vis_prong_p4
        mask = tau_vis_prong_mask
    elif mode == "prong_only":
        values = tau_vis_prong_only_p4
        mask = tau_vis_prong_only_mask
    elif mode == "rho":
        values = tau_vis_rho_p4
        mask = tau_vis_rho_mask
    else:
        raise ValueError(f"Unsupported visible tau mode '{mode}'.")

    if observable == "energy":
        return ak.to_numpy(ak.flatten(values.E[mask], axis=None), allow_missing=False).astype(np.float32)
    if observable == "pt":
        return ak.to_numpy(ak.flatten(values.pt[mask], axis=None), allow_missing=False).astype(np.float32)
    if observable == "eta":
        return ak.to_numpy(ak.flatten(values.eta[mask], axis=None), allow_missing=False).astype(np.float32)
    if observable == "phi":
        return ak.to_numpy(ak.flatten(values.phi[mask], axis=None), allow_missing=False).astype(np.float32)
    if observable == "mass":
        return ak.to_numpy(ak.flatten(values.mass[mask], axis=None), allow_missing=False).astype(np.float32)
    raise ValueError(f"Unsupported visible tau observable '{observable}'.")


def extract_part_feature(events: ak.Array, field_name: str) -> np.ndarray:
    if field_name not in events.fields:
        return np.array([], dtype=np.float32)
    values = ak.fill_none(filter_part_values(events, events[field_name]), np.nan)
    return ak.to_numpy(ak.flatten(values, axis=None), allow_missing=False).astype(np.float32)


def extract_part_momentum_observable(events: ak.Array, observable: str) -> np.ndarray:
    part_p4 = build_filtered_part_momentum4d(events)
    if observable == "energy":
        values = part_p4.E
    elif observable == "pt":
        values = part_p4.pt
    elif observable == "eta":
        values = ak.where(np.isfinite(part_p4.eta), part_p4.eta, np.nan)
    elif observable == "phi":
        values = part_p4.phi
    else:
        raise ValueError(f"Unsupported part momentum observable '{observable}'.")
    return ak.to_numpy(ak.flatten(values, axis=None), allow_missing=False).astype(np.float32)
