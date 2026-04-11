from __future__ import annotations

import awkward as ak
import numpy as np
import vector


PHOTON_DR_MAX = 0.3
FOUR_VECTOR_FEATURES = ["energy", "pt", "eta", "phi"]
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


def compute_pt_eta_phi(px: ak.Array, py: ak.Array, pz: ak.Array) -> tuple[ak.Array, ak.Array, ak.Array]:
    p4 = build_momentum4d(px, py, pz, np.sqrt(px * px + py * py + pz * pz))
    return p4.pt, p4.eta, p4.phi


def features_from_p4(p4):
    if isinstance(p4, ak.Array):
        eta = ak.where(np.isfinite(p4.eta), p4.eta, 0.0)
        phi = ak.where(np.isfinite(p4.phi), p4.phi, 0.0)
        return ak.concatenate(
            [
                ak.values_astype(p4.E, np.float32)[..., np.newaxis],
                ak.values_astype(p4.pt, np.float32)[..., np.newaxis],
                ak.values_astype(eta, np.float32)[..., np.newaxis],
                ak.values_astype(phi, np.float32)[..., np.newaxis],
            ],
            axis=-1,
        )

    eta = np.asarray(p4.eta, dtype=np.float32)
    phi = np.asarray(p4.phi, dtype=np.float32)
    eta = np.where(np.isfinite(eta), eta, 0.0)
    phi = np.where(np.isfinite(phi), phi, 0.0)
    return np.stack(
        [
            np.asarray(p4.E, dtype=np.float32),
            np.asarray(p4.pt, dtype=np.float32),
            eta,
            phi,
        ],
        axis=-1,
    ).astype(np.float32)


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
    return ak.sum(part_p4[mask], axis=1)


def map_hemisphere_to_tau_sign(first_values, second_values, first_charge, second_charge):
    first_is_tau_minus = (first_charge < 0) & (second_charge > 0)
    second_is_tau_minus = (second_charge < 0) & (first_charge > 0)
    tau_minus_mask = first_is_tau_minus | second_is_tau_minus
    tau_plus_mask = tau_minus_mask

    tau_minus = ak.where(first_is_tau_minus, first_values, second_values)
    tau_plus = ak.where(first_is_tau_minus, second_values, first_values)
    return tau_minus, tau_plus, tau_minus_mask, tau_plus_mask


def build_visible_tau_assumption_p4(events: ak.Array):
    charge = events["Part_charge"]
    hemisphere = events["Part_hemisphere"]
    pdg_id = abs(events["Part_pdgId"])
    part_p4 = build_part_momentum4d(events)

    hemisphere_masks = {
        "a": hemisphere == 1,
        "b": hemisphere == -1,
    }

    prong_charge_sums = {}
    prong_p4_by_hemisphere = {}
    rho_p4_by_hemisphere = {}

    for hemisphere_name, hemisphere_mask in hemisphere_masks.items():
        prong_mask = hemisphere_mask & (charge != 0)
        photon_mask = hemisphere_mask & (charge == 0) & (pdg_id == 21)
        prong_p4_constituents = part_p4[prong_mask]
        photon_p4_constituents = part_p4[photon_mask]

        prong_p4 = sum_masked_p4(events, prong_mask)
        prong_p4_by_hemisphere[hemisphere_name] = prong_p4
        prong_charge_sums[hemisphere_name] = ak.values_astype(ak.sum(charge[prong_mask], axis=1), np.float32)

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

        photon_p4 = ak.sum(photon_p4_constituents[photon_near_prong], axis=1)
        rho_p4_by_hemisphere[hemisphere_name] = prong_p4 + photon_p4

    prong_tau_minus, prong_tau_plus, prong_tau_minus_mask, prong_tau_plus_mask = map_hemisphere_to_tau_sign(
        prong_p4_by_hemisphere["a"],
        prong_p4_by_hemisphere["b"],
        prong_charge_sums["a"],
        prong_charge_sums["b"],
    )
    rho_tau_minus, rho_tau_plus, rho_tau_minus_mask, rho_tau_plus_mask = map_hemisphere_to_tau_sign(
        rho_p4_by_hemisphere["a"],
        rho_p4_by_hemisphere["b"],
        prong_charge_sums["a"],
        prong_charge_sums["b"],
    )
    return (
        stack_tau_pair(prong_tau_minus, prong_tau_plus),
        stack_tau_pair_mask(prong_tau_minus_mask, prong_tau_plus_mask),
        stack_tau_pair(rho_tau_minus, rho_tau_plus),
        stack_tau_pair_mask(rho_tau_minus_mask, rho_tau_plus_mask),
    )


def build_visible_tau_assumptions(events: ak.Array):
    # Shape convention is [event, tau(2)] with tau order [tau-, tau+].
    return build_visible_tau_assumption_p4(events)


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
):
    n_events = len(events)
    x_invisible_p4 = zero_p4((n_events, 2))
    x_invisible_mask = ak.Array(np.zeros((n_events, 2), dtype=bool))
    num_invisible_raw = np.zeros(n_events, dtype=np.int64)
    num_invisible_valid = np.zeros(n_events, dtype=np.int64)
    tau_vis_target_p4 = zero_p4((n_events, 2))
    tau_vis_target_mask = ak.Array(np.zeros((n_events, 2), dtype=bool))

    required_fields = [
        "truth_tau_px",
        "truth_tau_py",
        "truth_tau_pz",
        "truth_tau_E",
        "truth_anti_tau_px",
        "truth_anti_tau_py",
        "truth_anti_tau_pz",
        "truth_anti_tau_E",
        "event_category",
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
    event_category = ak.values_astype(events["event_category"], np.int64)
    tau_minus_category = event_category % 10
    tau_plus_category = event_category // 10

    tau_minus_use_rho = tau_minus_category == 2
    tau_plus_use_rho = tau_plus_category == 2

    tau_minus_target_p4 = ak.where(tau_minus_use_rho, tau_vis_rho_p4[:, 0], tau_vis_prong_p4[:, 0])
    tau_plus_target_p4 = ak.where(tau_plus_use_rho, tau_vis_rho_p4[:, 1], tau_vis_prong_p4[:, 1])
    tau_minus_target_mask = ak.where(tau_minus_use_rho, tau_vis_rho_mask[:, 0], tau_vis_prong_mask[:, 0])
    tau_plus_target_mask = ak.where(tau_plus_use_rho, tau_vis_rho_mask[:, 1], tau_vis_prong_mask[:, 1])
    tau_vis_target_p4 = stack_tau_pair(tau_minus_target_p4, tau_plus_target_p4)
    tau_vis_target_mask = stack_tau_pair_mask(tau_minus_target_mask, tau_plus_target_mask)

    tau_truth_valid = np.isfinite(truth_tau.E) & np.isfinite(truth_tau.px) & np.isfinite(truth_tau.py) & np.isfinite(truth_tau.pz)
    anti_tau_truth_valid = np.isfinite(truth_anti_tau.E) & np.isfinite(truth_anti_tau.px) & np.isfinite(truth_anti_tau.py) & np.isfinite(truth_anti_tau.pz)
    x_invisible_p4 = stack_tau_pair(truth_tau, truth_anti_tau)
    x_invisible_minus_mask = tau_truth_valid
    x_invisible_plus_mask = anti_tau_truth_valid
    x_invisible_mask = ak.concatenate([x_invisible_minus_mask[:, np.newaxis], x_invisible_plus_mask[:, np.newaxis]], axis=1)
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

    if observable == "energy":
        return ak.to_numpy(ak.flatten(x_invisible_p4.E[x_invisible_mask], axis=None), allow_missing=False).astype(np.float32)
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

    if mode == "prong":
        values = tau_vis_prong_p4
        mask = tau_vis_prong_mask
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
    values = ak.fill_none(events[field_name], np.nan)
    return ak.to_numpy(ak.flatten(values, axis=None), allow_missing=False).astype(np.float32)


def extract_part_momentum_observable(events: ak.Array, observable: str) -> np.ndarray:
    part_p4 = build_part_momentum4d(events)
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
